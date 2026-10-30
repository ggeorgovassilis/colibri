/*
 * pipeline_serve.c -- a mux-protocol server that runs a 2-stage Segment
 * pipeline over the Colibri Edge + Segment runtimes.
 *
 * Why a new binary instead of wiring the overlap into qwen36.c's serve_mux:
 * the main engine's serve path drives one synchronous forward through every
 * layer over one shared Model (shared falloc allocator, shared mux_rows,
 * token_count and process-global expert tier). Making two layer ranges run
 * concurrently inside that path would mean surgery on shared state. The
 * Segment runtime already gives us per-session isolated state and per-engine
 * run mutexes -- exactly what the proven 2-stage ring (tests/test_pipeline_overlap.c)
 * relies on -- so this binary reuses it unchanged.
 *
 * Pipeline: edge (tokenize/embed/logits, driver thread only) -> stage0
 * [0, split) -> stage1 [split, n) -> edge. Two worker threads own the two
 * stage engines; the driver keeps one slot in flight per stage each tick, so
 * stage-1 of request A overlaps stage-0 of request B. With one active request
 * the ring alternates (correct, serial); with >= 2 it overlaps.
 *
 * Transport: the same serve_codec.h frames the gateway speaks (READY/CAPS/
 * STAT, ACCEPT, DATA, DONE, ERROR, STOP, CANCEL). Sampling is qwen36_sample.h
 * over the full logits from coli_edge_logits (greedy for temp<=0).
 *
 * Launch: SNAP=<model dir> [SERVE=1] KV_SLOTS=<1..16> QT_LAYER_SPLIT=<split>
 *   [COLI_GPUS=0,1 QT_HOME=layer] ./pipeline_serve[_cuda]
 * The CUDA build places stage0 on device 0 and stage1 on device 1 when the
 * tier contract COLI_GPUS=0,1 + QT_HOME=layer + QT_LAYER_SPLIT is set; it
 * falls back to the CPU ranges when it declines.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
#endif

/* compat.h's inline meminfo reader uses stdio; the standard headers above must
 * precede it (the engine .c files all include <stdio.h> first for this reason). */
#include "compat.h"
#include "serve_codec.h"
#include "serve_poll.h"

#include "edge_adapters.h"
#include "edge_runtime.h"
#include "segment_adapters.h"
#include "segment_runtime.h"
#include "qwen36_sample.h"

/* ---- wall clock + rss --------------------------------------------------- */

static double now_ms(void) {
#ifdef _WIN32
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static double rss_gb(void) {
#if defined(_WIN32)
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1048576.0;      /* compat shim returns KiB */
#elif defined(__APPLE__)
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1073741824.0;   /* bytes on macOS */
#else
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1048576.0;      /* KiB on Linux */
#endif
}

/* Incremental UTF-8 buffering, copied verbatim from qwen36.c so DATA frames
 * never split a codepoint across token boundaries. */
static int utf8_drain(unsigned char *buf, int *bn, const unsigned char *b,
                      int n, unsigned char *out, int *outn) {
    *outn = 0;
    for (int k = 0; k < n; k++) { if (*bn < 16) buf[(*bn)++] = b[k]; }
    int j = 0;
    while (j < *bn) {
        unsigned char lead = buf[j]; int need;
        if (lead < 0x80) need = 1;
        else if ((lead & 0xE0) == 0xC0) need = 2;
        else if ((lead & 0xF0) == 0xE0) need = 3;
        else if ((lead & 0xF8) == 0xF0) need = 4;
        else { memmove(buf + j, buf + j + 1, *bn - j - 1); (*bn)--; continue; }
        if (j + need > *bn) break;
        if (*outn + need <= 255) { for (int x = 0; x < need; x++) out[(*outn)++] = buf[j + x]; }
        memmove(buf + j, buf + j + need, *bn - j - need);
        *bn -= need;
    }
    return *outn;
}

/* ---- reusable barrier (POSIX, as in tests/test_pipeline_overlap.c) ------- */

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int nparties;
    int arrived;
    int generation;
} barrier_t;

static void barrier_init(barrier_t *b, int nparties) {
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->nparties = nparties;
    b->arrived = 0;
    b->generation = 0;
}

static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->mu);
    int generation = b->generation;
    b->arrived++;
    if (b->arrived == b->nparties) {
        b->arrived = 0;
        b->generation++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (generation == b->generation)
            pthread_cond_wait(&b->cv, &b->mu);
    }
    pthread_mutex_unlock(&b->mu);
}

static void barrier_destroy(barrier_t *b) {
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
}

/* ---- global engine + serving state -------------------------------------- */

static const char *g_model_dir;
static ColiEdgeEngine *g_edge;
static ColiSegmentEngine *g_s0_engine;
static ColiSegmentEngine *g_s1_engine;
static ColiEdgeCapabilities g_cap;

static size_t g_row_bytes;      /* state_width * sizeof(float) */
static int g_vocab;
static uint32_t g_max_ctx;      /* engine context ceiling */
static uint32_t g_max_batch;    /* max rows per embed/run call */
static int g_split;             /* stage0 = [0, split), stage1 = [split, n) */
static int g_kv_slots;

static int32_t g_eos[16];
static int g_n_eos;

static const ColiServeWireProfile g_wire = {
    .max_header_bytes = 4096,
    .max_payload_bytes = 1u << 24,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
};

/* ---- per-request state -------------------------------------------------- */

typedef struct {
    int active;
    int cancel;
    int stop;
    char id[COLI_SERVE_ID_CAP];
    int slot;
    int max_tok;
    int logprobs;
    float temp, top_p;
    int prompt_tokens;
    int32_t *ids;           /* prompt token ids (freed at finish) */

    ColiSegmentSession *s0;
    ColiSegmentSession *s1;
    float *emb;             /* embedding of the next stage-0 token */
    float *inter;           /* stage-0 output, in flight to stage-1 */
    float *out;             /* stage-1 output (pre-head hidden state) */
    float *logits;          /* vocab floats */
    int32_t token;          /* token currently embedded in `emb` */
    int32_t inter_token;    /* token whose hidden state is in `inter` */
    uint64_t pos;           /* position of the next stage-0 token */
    uint64_t inter_pos;     /* position of the token in `inter` */
    int inter_pending;      /* `inter` holds a token for stage-1 */

    int n_gen;              /* generated tokens so far */
    int length_limited;     /* stopped at max_tok, not EOS */
    double t0;              /* clock when the turn started (ms) */

    unsigned char sbuf[16]; /* UTF-8 carry */
    int sbn;
    char error[512];
} PipeReq;

static PipeReq *g_req;

/* FIFO queues of slot indices ready for each stage. */
typedef struct {
    int *buf;
    int cap;
    int head;
    int n;
} IntQueue;

static IntQueue g_s0q, g_s1q;

static void iq_init(IntQueue *q, int cap) {
    q->buf = malloc((size_t)cap * sizeof(int));
    q->cap = cap;
    q->head = 0;
    q->n = 0;
}

static void iq_push(IntQueue *q, int v) {
    if (q->n >= q->cap) return;           /* invariant: a slot is in <= 1 queue */
    q->buf[(q->head + q->n) % q->cap] = v;
    q->n++;
}

static int iq_pop(IntQueue *q) {
    if (q->n <= 0) return -1;
    int v = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    q->n--;
    return v;
}

/* ---- segment / edge primitives ------------------------------------------ */

static int run_stage(ColiSegmentSession *session, uint32_t rows,
                     uint64_t position, const int32_t *token_ids,
                     size_t token_count, const float *input, float *output,
                     size_t row_bytes, char *error, size_t error_size) {
    ColiSegmentRunRequest run = {
        .struct_size = sizeof(run), .rows = rows, .position = position,
        .token_ids = token_ids, .token_count = token_count,
        .input = input, .input_bytes = (size_t)rows * row_bytes,
        .output = output, .output_bytes = (size_t)rows * row_bytes,
    };
    return coli_segment_run(session, &run, error, error_size);
}

static int embed_tokens(ColiEdgeEngine *edge, const int32_t *tokens,
                        size_t count, float *output, size_t row_bytes,
                        char *error, size_t error_size) {
    ColiEdgeEmbedRequest embed = {
        .struct_size = sizeof(embed), .rows = (uint32_t)count,
        .token_ids = tokens, .token_count = count,
        .output = output, .output_bytes = count * row_bytes,
    };
    return coli_edge_embed(edge, &embed, error, error_size);
}

static int logits_row(ColiEdgeEngine *edge, const float *input,
                      size_t row_bytes, float *logits, int vocab,
                      char *error, size_t error_size) {
    ColiEdgeLogitsRequest lr = {
        .struct_size = sizeof(lr), .rows = 1, .input = input,
        .input_bytes = row_bytes, .logits = logits,
        .logits_capacity = (size_t)vocab,
    };
    return coli_edge_logits(edge, &lr, error, error_size);
}

static int tokenize(ColiEdgeEngine *edge, const char *text, size_t length,
                    int32_t **ids, int *count, char *error, size_t error_size) {
    size_t token_count = 0;
    if (coli_edge_tokenize(edge, text, length, NULL, 0, &token_count,
                           error, error_size) != 0 || !token_count)
        return -1;
    int32_t *tokens = malloc(token_count * sizeof(*tokens));
    if (!tokens) return -1;
    if (coli_edge_tokenize(edge, text, length, tokens, token_count,
                           &token_count, error, error_size) != 0) {
        free(tokens);
        return -1;
    }
    *ids = tokens;
    *count = (int)token_count;
    return 0;
}

/* ---- engine/session setup ------------------------------------------------ */

static int open_stage(ColiSegmentEngine **engine, uint32_t begin, uint32_t end,
                      char *error, size_t error_size) {
    ColiSegmentEngineOptions options = {
        .struct_size = sizeof(options), .model_dir = g_model_dir,
        .layer_begin = begin, .layer_end = end, .context_tokens = g_max_ctx,
        /* request the CUDA-backed range when this binary links it; the engine
         * falls back to the CPU range internally when it declines */
        .backend_mask = COLI_SEGMENT_CAP_CPU | COLI_SEGMENT_CAP_CUDA,
    };
    if (coli_segment_engine_open("qwen36", &options, engine,
                                 error, error_size) != 0) {
        options.backend_mask = COLI_SEGMENT_CAP_CPU;
        if (coli_segment_engine_open("qwen36", &options, engine,
                                     error, error_size) != 0)
            return -1;
    }
    return 0;
}

static int create_session(ColiSegmentEngine *engine, ColiSegmentSession **session,
                          uint32_t context, char *error, size_t error_size) {
    ColiSegmentSessionOptions options = {
        .struct_size = sizeof(options), .context_tokens = context,
    };
    return coli_segment_session_create(engine, &options, session,
                                       error, error_size);
}

static int is_eos(int tok) {
    for (int i = 0; i < g_n_eos; i++)
        if (tok == g_eos[i]) return 1;
    return 0;
}

static void resolve_eos(void) {
    g_n_eos = 0;
    const char *ov = getenv("Q36_EOS");
    if (ov && *ov) { g_eos[g_n_eos++] = (int32_t)atoi(ov); return; }
    if (g_cap.eos_token_id >= 0)
        g_eos[g_n_eos++] = g_cap.eos_token_id;
    char error[256];
    int32_t tmp[8];
    size_t tn = 0;
    const char *specials[] = { "<|im_end|>", "<|endoftext|>" };
    for (int s = 0; s < 2 && g_n_eos < 16; s++) {
        if (coli_edge_tokenize(g_edge, specials[s], strlen(specials[s]),
                               tmp, 8, &tn, error, sizeof(error)) == 0 &&
            tn == 1)
            g_eos[g_n_eos++] = tmp[0];
    }
    if (!g_n_eos)
        g_eos[g_n_eos++] = 151645;   /* tokenizer without added_tokens: old default */
}

/* ---- finish / emit ------------------------------------------------------- */

static void finish_request(int slot, int cancelled, int errored) {
    PipeReq *q = &g_req[slot];
    if (!q->active) return;
    if (cancelled) {
        coli_serve_write_error(stdout, q->id, "CANCELLED");
    } else if (errored) {
        coli_serve_write_error(stdout, q->id, q->error[0] ? q->error : "pipeline worker failed");
    } else {
        if (q->sbn > 0)
            coli_serve_write_data(stdout, q->id, q->sbuf, (size_t)q->sbn);
        double secs = (now_ms() - q->t0) / 1000.0;
        ColiServeDone done = {
            .completion_tokens = q->n_gen,
            .tokens_per_second = secs > 0.0 ? (double)q->n_gen / secs : 0.0,
            .cache_hit_percent = 0.0,
            .rss_gb = rss_gb(),
            .prompt_tokens = q->prompt_tokens,
            .length_limited = q->length_limited,
        };
        coli_serve_write_done(stdout, q->id, &done);
    }
    if (q->s0) coli_segment_session_destroy(q->s0);
    if (q->s1) coli_segment_session_destroy(q->s1);
    free(q->emb);
    free(q->inter);
    free(q->out);
    free(q->logits);
    free(q->ids);
    memset(q, 0, sizeof(*q));
}

static void emit_token(PipeReq *q, int tok) {
    unsigned char tmp[256];
    size_t tn = 0;
    char error[256];
    if (coli_edge_detokenize(g_edge, (const int32_t[]){ (int32_t)tok }, 1,
                             (char *)tmp, sizeof(tmp), &tn,
                             error, sizeof(error)) != 0)
        return;
    if (q->logprobs > 0) {
        char lptail[1024];
        coli_logprob_tail(lptail, sizeof(lptail), q->logits, g_vocab, tok,
                          q->logprobs);
        coli_serve_write_data_lp(stdout, q->id, tmp, tn, lptail);
    } else {
        unsigned char chunk[256];
        int cn = 0;
        utf8_drain(q->sbuf, &q->sbn, tmp, (int)tn, chunk, &cn);
        if (cn > 0)
            coli_serve_write_data(stdout, q->id, chunk, (size_t)cn);
    }
}

/* ---- pipeline ring ------------------------------------------------------- */

typedef struct {
    barrier_t barrier;      /* 3 parties: driver + 2 stage workers */
    int s0_req, s1_req;     /* slot each stage runs, or -1 */
    int done;
} Ring;

static Ring g_ring;

static void *stage0_worker(void *arg) {
    Ring *r = arg;
    for (;;) {
        barrier_wait(&r->barrier);
        if (r->done) break;
        if (r->s0_req >= 0) {
            PipeReq *q = &g_req[r->s0_req];
            int rc = run_stage(q->s0, 1, q->pos, &q->token, 1, q->emb, q->inter,
                               g_row_bytes, q->error, sizeof(q->error));
            if (rc != 0) {
                /* q->error already carries the adapter's message */
            } else {
                q->inter_token = q->token;
                q->inter_pos = q->pos;
                q->pos++;
                q->inter_pending = 1;
            }
        }
        barrier_wait(&r->barrier);
    }
    return NULL;
}

static void *stage1_worker(void *arg) {
    Ring *r = arg;
    for (;;) {
        barrier_wait(&r->barrier);
        if (r->done) break;
        if (r->s1_req >= 0) {
            PipeReq *q = &g_req[r->s1_req];
            int rc = run_stage(q->s1, 1, q->inter_pos, &q->inter_token, 1,
                               q->inter, q->out, g_row_bytes,
                               q->error, sizeof(q->error));
            (void)rc;
        }
        barrier_wait(&r->barrier);
    }
    return NULL;
}

/* Pop the next runnable slot, finishing any that were stopped or cancelled
 * in the meantime. */
static int pop_ready(IntQueue *q) {
    for (;;) {
        int slot = iq_pop(q);
        if (slot < 0) return -1;
        PipeReq *r = &g_req[slot];
        if (r->cancel) { finish_request(slot, 1, 0); continue; }
        if (r->stop) { finish_request(slot, 0, 0); continue; }
        return slot;
    }
}

/* Stage-1 produced `out` for a token: sample, emit, and either feed the next
 * token back to stage-0 or finish the request. */
static void advance_s1(int slot) {
    PipeReq *q = &g_req[slot];
    q->inter_pending = 0;
    if (logits_row(g_edge, q->out, g_row_bytes, q->logits, g_vocab,
                   q->error, sizeof(q->error)) != 0) {
        finish_request(slot, 0, 1);
        return;
    }
    int tok = serve_sample(q->logits, g_vocab, q->temp, q->top_p);
    if (is_eos(tok)) {
        q->length_limited = 0;
        finish_request(slot, 0, 0);
        return;
    }
    emit_token(q, tok);
    q->n_gen++;
    if (q->n_gen >= q->max_tok) {
        q->length_limited = 1;
        finish_request(slot, 0, 0);
        return;
    }
    q->token = (int32_t)tok;
    if (embed_tokens(g_edge, &q->token, 1, q->emb, g_row_bytes,
                     q->error, sizeof(q->error)) != 0) {
        finish_request(slot, 0, 1);
        return;
    }
    iq_push(&g_s0q, slot);
}

static void tick_pipeline(void) {
    Ring *r = &g_ring;
    r->s0_req = pop_ready(&g_s0q);
    r->s1_req = pop_ready(&g_s1q);
    barrier_wait(&r->barrier);   /* release the two stages */
    barrier_wait(&r->barrier);   /* wait for both stages to finish */
    if (r->s0_req >= 0) {
        PipeReq *q = &g_req[r->s0_req];
        if (q->error[0])
            finish_request(r->s0_req, 0, 1);
        else
            iq_push(&g_s1q, r->s0_req);
    }
    if (r->s1_req >= 0) {
        PipeReq *q = &g_req[r->s1_req];
        if (q->error[0])
            finish_request(r->s1_req, 0, 1);
        else
            advance_s1(r->s1_req);
    }
}

/* ---- command handling ---------------------------------------------------- */

static void handle_submit(ColiServeCommand *cmd) {
    int slot = cmd->slot;
    if (slot < 0 || slot >= g_kv_slots) {
        coli_serve_write_error(stdout, cmd->id, "invalid cache slot");
        return;
    }
    if (g_req[slot].active) {
        coli_serve_write_error(stdout, cmd->id, "SLOT_BUSY");
        return;
    }
    char error[512];
    int32_t *ids = NULL;
    int np = 0;
    if (tokenize(g_edge, (const char *)cmd->payload, (size_t)cmd->payload_bytes,
                 &ids, &np, error, sizeof(error)) != 0 || np < 1) {
        free(ids);
        coli_serve_write_error(stdout, cmd->id, "EMPTY_PROMPT");
        return;
    }
    if ((uint64_t)np >= g_max_ctx) {
        free(ids);
        coli_serve_write_error(stdout, cmd->id, "CONTEXT_EXCEEDED");
        return;
    }
    int budget = cmd->max_tokens;
    if (budget > (int)(g_max_ctx - (uint64_t)np))
        budget = (int)(g_max_ctx - (uint64_t)np);

    PipeReq *q = &g_req[slot];
    memset(q, 0, sizeof(*q));
    snprintf(q->id, sizeof(q->id), "%s", cmd->id);
    q->slot = slot;
    q->active = 1;
    q->max_tok = budget;
    q->temp = cmd->temperature;
    q->top_p = cmd->top_p;
    q->logprobs = cmd->logprobs;
    q->prompt_tokens = np;
    q->ids = ids;
    q->t0 = now_ms();

    uint32_t ctx = (uint32_t)(np + budget + 2);
    if (ctx > g_max_ctx) ctx = g_max_ctx;
    if (create_session(g_s0_engine, &q->s0, ctx, error, sizeof(error)) != 0 ||
        create_session(g_s1_engine, &q->s1, ctx, error, sizeof(error)) != 0) {
        snprintf(q->error, sizeof(q->error), "%s", error);
        finish_request(slot, 0, 1);
        return;
    }
    q->emb = malloc(g_row_bytes);
    q->inter = malloc(g_row_bytes);
    q->out = malloc(g_row_bytes);
    q->logits = malloc((size_t)g_vocab * sizeof(float));
    if (!q->emb || !q->inter || !q->out || !q->logits) {
        snprintf(q->error, sizeof(q->error), "out of memory for request state");
        finish_request(slot, 0, 1);
        return;
    }

    /* prefill: embed the prompt -> stage0 -> stage1 -> logits at the last
     * prompt position. Each component caps how many rows it accepts per call
     * (the qwen36 adapters: 128), so a long prompt is pre-filled in chunks of
     * g_max_batch rows. This runs in the driver thread between ticks, while
     * both workers idle at the barrier, so the per-engine run mutexes are
     * uncontended. */
    size_t width = g_row_bytes / sizeof(float);
    size_t cap = (size_t)g_max_batch * g_row_bytes;
    float *input = malloc(cap), *inter = malloc(cap), *out = malloc(cap);
    if (!input || !inter || !out) {
        snprintf(q->error, sizeof(q->error), "out of memory for prefill");
        free(input); free(inter); free(out);
        finish_request(slot, 0, 1);
        return;
    }
    int fail = 0;
    size_t last_rows = 0;
    for (size_t pos = 0; pos < (size_t)np; ) {
        size_t rows = (size_t)np - pos;
        if (rows > g_max_batch) rows = g_max_batch;
        if (embed_tokens(g_edge, ids + pos, rows, input, g_row_bytes,
                         error, sizeof(error)) != 0 ||
            run_stage(q->s0, (uint32_t)rows, pos, ids + pos, rows, input, inter,
                      g_row_bytes, error, sizeof(error)) != 0 ||
            run_stage(q->s1, (uint32_t)rows, pos, ids + pos, rows, inter, out,
                      g_row_bytes, error, sizeof(error)) != 0) {
            fail = 1;
            break;
        }
        last_rows = rows;
        pos += rows;
    }
    if (!fail)
        fail = logits_row(g_edge, out + (last_rows - 1) * width,
                          g_row_bytes, q->logits, g_vocab,
                          error, sizeof(error)) != 0;
    free(out); free(inter); free(input);
    if (fail) {
        snprintf(q->error, sizeof(q->error), "%s", error);
        finish_request(slot, 0, 1);
        return;
    }
    q->pos = (uint64_t)np;

    coli_serve_write_accept(stdout, q->id, np);

    if (budget < 1) {   /* read-only logprobs request: prefill done, nothing to generate */
        finish_request(slot, 0, 0);
        return;
    }

    int tok = serve_sample(q->logits, g_vocab, q->temp, q->top_p);
    if (is_eos(tok)) {
        q->length_limited = 0;
        finish_request(slot, 0, 0);
        return;
    }
    emit_token(q, tok);
    q->n_gen = 1;
    if (q->n_gen >= q->max_tok) {
        q->length_limited = 1;
        finish_request(slot, 0, 0);
        return;
    }
    q->token = (int32_t)tok;
    if (embed_tokens(g_edge, &q->token, 1, q->emb, g_row_bytes,
                     q->error, sizeof(q->error)) != 0) {
        finish_request(slot, 0, 1);
        return;
    }
    iq_push(&g_s0q, slot);
}

static void handle_control(ColiServeCommand *cmd) {
    for (int i = 0; i < g_kv_slots; i++) {
        if (g_req[i].active && !strcmp(g_req[i].id, cmd->id)) {
            if (cmd->kind == COLI_SERVE_COMMAND_CANCEL)
                g_req[i].cancel = 1;
            else if (cmd->kind == COLI_SERVE_COMMAND_STOP)
                g_req[i].stop = 1;
            return;
        }
    }
}

/* ---- main ---------------------------------------------------------------- */

int main(int argc, char **argv) {
    g_model_dir = getenv("SNAP");
    if (!g_model_dir || !*g_model_dir)
        g_model_dir = argc > 1 ? argv[1] : NULL;
    if (!g_model_dir || !*g_model_dir) {
        fprintf(stderr, "usage: SNAP=<model directory> %s\n", argv[0]);
        return 2;
    }

    const char *ks = getenv("KV_SLOTS");
    g_kv_slots = ks && *ks ? atoi(ks) : 4;
    if (g_kv_slots < 1) g_kv_slots = 1;
    if (g_kv_slots > 16) g_kv_slots = 16;

    char error[512] = {0};

    if (coli_qwen36_segment_adapter_register() != 0 ||
        coli_qwen36_edge_adapter_register() != 0) {
        fprintf(stderr, "[pipeline] cannot register the qwen36 adapters\n");
        return 1;
    }

    ColiEdgeEngineOptions edge_options = {
        .struct_size = sizeof(edge_options), .model_dir = g_model_dir,
    };
    if (coli_edge_engine_open("qwen36", &edge_options, &g_edge,
                              error, sizeof(error)) != 0) {
        fprintf(stderr, "[pipeline] edge: %s\n", error);
        return 1;
    }
    g_cap.struct_size = sizeof(g_cap);
    if (coli_edge_engine_capabilities(g_edge, &g_cap, error, sizeof(error)) != 0) {
        fprintf(stderr, "[pipeline] edge capabilities: %s\n", error);
        return 1;
    }
    if (g_cap.state_dtype != COLI_EDGE_DTYPE_F32 ||
        !(g_cap.flags & COLI_EDGE_CAP_LOGITS)) {
        fprintf(stderr, "[pipeline] edge adapter must expose F32 state and logits\n");
        return 1;
    }
    g_row_bytes = (size_t)g_cap.state_width * sizeof(float);
    g_vocab = (int)g_cap.vocab_size;
    g_max_ctx = g_cap.max_context_tokens ? g_cap.max_context_tokens : 262144;
    if (!g_max_ctx) g_max_ctx = 262144;

    g_split = g_cap.num_layers / 2;
    const char *sp = getenv("QT_LAYER_SPLIT");
    if (sp && *sp) g_split = atoi(sp);
    if (g_split < 1) g_split = 1;
    if (g_split >= (int)g_cap.num_layers) g_split = (int)g_cap.num_layers - 1;
    if (g_cap.num_layers < 2) {
        fprintf(stderr, "[pipeline] model has too few layers to split\n");
        return 1;
    }

    if (open_stage(&g_s0_engine, 0, (uint32_t)g_split, error, sizeof(error)) != 0) {
        fprintf(stderr, "[pipeline] stage0: %s\n", error);
        return 1;
    }
    if (open_stage(&g_s1_engine, (uint32_t)g_split, g_cap.num_layers,
                   error, sizeof(error)) != 0) {
        fprintf(stderr, "[pipeline] stage1: %s\n", error);
        return 1;
    }

    /* A prefill must never hand any component more rows than it accepts in
     * one call. The qwen36 edge/segment adapters cap a call at 128 rows, so
     * long prompts are pre-filled in chunks of that size (or less). */
    g_max_batch = g_cap.max_batch_rows ? g_cap.max_batch_rows : 128;
    ColiSegmentCapabilities s0cap, s1cap;
    s0cap.struct_size = sizeof(s0cap);
    s1cap.struct_size = sizeof(s1cap);
    if (coli_segment_engine_capabilities(g_s0_engine, &s0cap,
                                         error, sizeof(error)) == 0 &&
        s0cap.max_batch_rows && s0cap.max_batch_rows < g_max_batch)
        g_max_batch = s0cap.max_batch_rows;
    if (coli_segment_engine_capabilities(g_s1_engine, &s1cap,
                                         error, sizeof(error)) == 0 &&
        s1cap.max_batch_rows && s1cap.max_batch_rows < g_max_batch)
        g_max_batch = s1cap.max_batch_rows;
    if (!g_max_batch) g_max_batch = 128;

    resolve_eos();

    g_req = calloc((size_t)g_kv_slots, sizeof(*g_req));
    if (!g_req) {
        fprintf(stderr, "[pipeline] out of memory\n");
        return 1;
    }
    iq_init(&g_s0q, g_kv_slots);
    iq_init(&g_s1q, g_kv_slots);
    barrier_init(&g_ring.barrier, 3);

    pthread_t t0, t1;
    if (pthread_create(&t0, NULL, stage0_worker, &g_ring) != 0 ||
        pthread_create(&t1, NULL, stage1_worker, &g_ring) != 0) {
        fprintf(stderr, "[pipeline] cannot start pipeline workers\n");
        return 1;
    }

    coli_serve_stdio_init();
    fprintf(stderr, "[pipeline] qwen36 2-stage ring: layers [0,%d) + [%d,%u), "
            "KV_SLOTS=%d, ctx=%u\n",
            g_split, g_split, g_cap.num_layers, g_kv_slots, g_max_ctx);

    if (!coli_serve_write_ready_caps(stdout, rss_gb(), "vision=0")) {
        fprintf(stderr, "[pipeline] cannot write the handshake\n");
        return 1;
    }

    int input_eof = 0;
    for (;;) {
        int active = 0;
        for (int i = 0; i < g_kv_slots; i++) active += g_req[i].active;

        if (!input_eof && (!active || coli_serve_stdin_ready())) {
            ColiServeCommand cmd;
            ColiServeReadResult res = coli_serve_read_command(stdin, &g_wire, &cmd);
            if (res == COLI_SERVE_READ_EOF) {
                input_eof = 1;
            } else if (res == COLI_SERVE_READ_OK) {
                switch (cmd.kind) {
                case COLI_SERVE_COMMAND_SUBMIT:
                    handle_submit(&cmd);
                    break;
                case COLI_SERVE_COMMAND_STOP:
                case COLI_SERVE_COMMAND_CANCEL:
                    handle_control(&cmd);
                    break;
                case COLI_SERVE_COMMAND_IMAGE:
                    coli_serve_write_error(stdout, cmd.id,
                                           "this engine has no vision tower");
                    break;
                case COLI_SERVE_COMMAND_DECIDE:
                    coli_serve_write_error(stdout, cmd.id,
                                           "this engine does not decide");
                    break;
                default:
                    break;
                }
                coli_serve_command_dispose(&cmd);
            } else if (res == COLI_SERVE_READ_BAD_REQUEST ||
                       res == COLI_SERVE_READ_BAD_FRAME) {
                coli_serve_write_error(stdout, cmd.id[0] ? cmd.id : "0", "BAD_REQUEST");
                coli_serve_command_dispose(&cmd);
            }
            /* IGNORED and NOMEM fall through; NOMEM is fatal-ish but rare. */
        }

        active = 0;
        for (int i = 0; i < g_kv_slots; i++) active += g_req[i].active;
        if (!active) {
            if (input_eof) break;
            continue;
        }
        tick_pipeline();
    }

    /* graceful shutdown: wake the workers once, then join */
    g_ring.done = 1;
    barrier_wait(&g_ring.barrier);
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);
    barrier_destroy(&g_ring.barrier);

    free(g_s0q.buf);
    free(g_s1q.buf);
    free(g_req);
    (void)coli_segment_engine_close(g_s1_engine, NULL, 0);
    (void)coli_segment_engine_close(g_s0_engine, NULL, 0);
    coli_edge_engine_close(g_edge);
    return 0;
}
