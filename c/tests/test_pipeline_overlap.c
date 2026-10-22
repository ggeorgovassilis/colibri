#include "../edge_adapters.h"
#include "../edge_runtime.h"
#include "../json.h"
#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * M1b pipeline overlap driver: run TWO concurrent greedy decodes through a
 * 2-stage Segment chain (edge -> stage[0, split) -> stage[split, n) -> edge)
 * with a lock-step ring. At every steady-state tick stage-0 advances one
 * request's next token while stage-1 drains the OTHER request's in-flight
 * activation, so the two engines execute on two threads concurrently. Each
 * stage engine owns its own run mutex, which is what makes the cross-engine
 * overlap safe without touching the process-global accelerator tier.
 *
 * The pipelined token streams are checked against a serial full-range decode
 * of each request (and against the independent oracle for the first request
 * when a reference JSON is supplied). Overlap is witnessed with an in-flight
 * counter: the two workers rendezvous before each forward, so the peak
 * concurrency must reach two. Wall-clock timings are reported for
 * observability; throughput is not the point of this PoC.
 *
 * M2 adds a timestamped timeline of every stage/driver slice and derives the
 * pipeline bubble ratio from it (0 = packed, 0.5 = serialised), so the overlap
 * is not only asserted but visible.
 *
 * usage: test_pipeline_overlap FAMILY MODEL_DIR SPLIT MAX_NEW [REF_JSON]
 *   REF_JSON may be "-" to skip the oracle and drive two fixed text prompts.
 */

#define REQUIRE(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "%s: %s\n", family, message); goto fail; } \
} while (0)

/* ---- timing ----------------------------------------------------------- */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* One timestamped activity in the pipeline timeline (M2 observability). */
typedef struct {
    int stage;   /* 0 or 1, or -1 for the driver's select/embed slice */
    int req;     /* request index the activity belongs to */
    double t0;   /* start, ms (monotonic) */
    double t1;   /* end, ms (monotonic) */
} TraceEvent;

/* ---- reusable barrier (POSIX, portable across the supported hosts) ---- */

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

/* ---- atomic in-flight witness (GCC/Clang builtins, no -std dependency) - */

static int inflight_add(int *p, int v) {
    return __atomic_add_fetch(p, v, __ATOMIC_SEQ_CST);
}
static int inflight_load(const int *p) {
    return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}
static void inflight_peak(int *peak, int value) {
    int cur = __atomic_load_n(peak, __ATOMIC_SEQ_CST);
    while (value > cur &&
           !__atomic_compare_exchange_n(peak, &cur, value, 0,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
}

/* ---- JSON / token-array helpers (mirrors test_pipeline_real.c) -------- */

static char *read_file(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (!stream) return NULL;
    if (fseek(stream, 0, SEEK_END)) { fclose(stream); return NULL; }
    long length = ftell(stream);
    if (length < 0 || fseek(stream, 0, SEEK_SET)) { fclose(stream); return NULL; }
    char *data = malloc((size_t)length + 1u);
    if (!data || fread(data, 1, (size_t)length, stream) != (size_t)length) {
        free(data); fclose(stream); return NULL;
    }
    data[length] = '\0'; fclose(stream);
    return data;
}

static int32_t *read_ids(jval *object, const char *key, int *count) {
    jval *array = json_get(object, key);
    if (!array || array->t != J_ARR || array->len < 1) return NULL;
    int32_t *ids = malloc((size_t)array->len * sizeof(*ids));
    if (!ids) return NULL;
    for (int item = 0; item < array->len; item++) {
        if (!array->kids[item] || array->kids[item]->t != J_NUM ||
            array->kids[item]->num < INT32_MIN ||
            array->kids[item]->num > INT32_MAX ||
            array->kids[item]->num != (double)(int32_t)array->kids[item]->num) {
            free(ids); return NULL;
        }
        ids[item] = (int32_t)array->kids[item]->num;
    }
    *count = array->len;
    return ids;
}

/* ---- segment / edge primitives ---------------------------------------- */

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

static int select_token(ColiEdgeEngine *edge, const float *logits,
                        size_t row_bytes, int32_t *token,
                        char *error, size_t error_size) {
    float score = 0.0f;
    ColiEdgeSelectRequest select = {
        .struct_size = sizeof(select), .rows = 1, .input = logits,
        .input_bytes = row_bytes, .token_ids = token, .token_capacity = 1,
        .scores = &score, .score_capacity = 1,
    };
    return coli_edge_select(edge, &select, error, error_size);
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
        free(tokens); return -1;
    }
    *ids = tokens;
    *count = (int)token_count;
    return 0;
}

/* ---- serial full-range reference decode -------------------------------- */

static int serial_decode(ColiSegmentSession *full, ColiEdgeEngine *edge,
                         size_t row_bytes, const int32_t *prompt,
                         int prompt_count, int max_new, int32_t *tokens,
                         char *error, size_t error_size) {
    size_t cap = (size_t)prompt_count * row_bytes;
    float *input = malloc(cap);
    float *output = malloc(cap);
    if (!input || !output) {
        free(output); free(input);
        snprintf(error, error_size, "out of memory for serial decode");
        return -1;
    }
    if (embed_tokens(edge, prompt, (size_t)prompt_count, input, row_bytes,
                     error, error_size) != 0)
        goto fail;
    if (run_stage(full, (uint32_t)prompt_count, 0, prompt,
                  (size_t)prompt_count, input, output, row_bytes,
                  error, error_size) != 0)
        goto fail;
    if (select_token(edge,
                     output + (size_t)(prompt_count - 1) *
                                  (row_bytes / sizeof(float)),
                     row_bytes, &tokens[0], error, error_size) != 0)
        goto fail;
    for (int g = 1; g < max_new; g++) {
        int32_t token = tokens[g - 1];
        if (embed_tokens(edge, &token, 1, input, row_bytes,
                         error, error_size) != 0)
            goto fail;
        if (run_stage(full, 1, (uint64_t)prompt_count + (uint64_t)g - 1u,
                      &token, 1, input, output, row_bytes,
                      error, error_size) != 0)
            goto fail;
        if (select_token(edge, output, row_bytes, &tokens[g],
                         error, error_size) != 0)
            goto fail;
    }
    free(output); free(input);
    return 0;
fail:
    free(output); free(input);
    return -1;
}

/* ---- per-request pipeline state --------------------------------------- */

typedef struct {
    ColiSegmentSession *s0; /* stage-0 session for this request */
    ColiSegmentSession *s1; /* stage-1 session for this request */
    float *emb;             /* embedding of the next stage-0 token */
    float *inter;           /* stage-0 output, in flight to stage-1 */
    float *out;             /* stage-1 output (pre-head logits) */
    int32_t token;          /* token currently embedded in `emb` */
    int32_t inter_token;    /* token whose hidden state is in `inter` */
    uint64_t pos;           /* position of the next stage-0 token */
    uint64_t inter_pos;     /* position of the token in `inter` */
    int inter_pending;      /* `inter` holds a token for stage-1 */
    int32_t *gen;           /* generated tokens (gen[0] from prefill) */
    int n_gen;
    char error[512];
} Request;

typedef struct {
    Request *req[2];
    ColiEdgeEngine *edge;
    size_t row_bytes;
    barrier_t barrier;      /* 3 parties: driver + 2 stage workers */
    int s0_req, s1_req;     /* request index each stage runs, or -1 */
    int done;
    int failed;
    int in_flight;
    int max_in_flight;
    /* M2 observability: timestamped timeline of every stage/driver slice. */
    TraceEvent *trace;
    int trace_cap;
    int trace_count;
    pthread_mutex_t trace_lock;
} Ring;

/* ---- M2 observability: timeline trace + bubble ratio ------------------ */

static void trace_record(Ring *r, int stage, int req, double t0, double t1) {
    pthread_mutex_lock(&r->trace_lock);
    if (r->trace_count < r->trace_cap)
        r->trace[r->trace_count++] = (TraceEvent){stage, req, t0, t1};
    pthread_mutex_unlock(&r->trace_lock);
}

/* Print the recorded timeline and report the pipeline bubble ratio: the
 * fraction of stage capacity idle across the run. 0 = perfectly packed,
 * 0.5 = the two stages never overlapped (serialised). Lower is better. */
static void trace_report(const Ring *r) {
    if (r->trace_count < 1) return;
    double tmin = r->trace[0].t0, tmax = r->trace[0].t1;
    double busy[2] = {0.0, 0.0};
    int runs[2] = {0, 0};
    for (int i = 0; i < r->trace_count; i++) {
        const TraceEvent *e = &r->trace[i];
        if (e->t0 < tmin) tmin = e->t0;
        if (e->t1 > tmax) tmax = e->t1;
        if (e->stage >= 0) {
            busy[e->stage] += e->t1 - e->t0;
            runs[e->stage]++;
        }
    }
    double makespan = tmax - tmin;
    double bubble = makespan > 0.0
        ? 1.0 - (busy[0] + busy[1]) / (2.0 * makespan) : 0.0;
    printf("  timeline (stage, req, t0, t1 in ms):\n");
    for (int i = 0; i < r->trace_count; i++) {
        const TraceEvent *e = &r->trace[i];
        const char *label = e->stage == 0 ? "S0" : e->stage == 1 ? "S1" : "DR";
        printf("    %s R%d  %8.2f %8.2f\n",
               label, e->req, e->t0 - tmin, e->t1 - tmin);
    }
    printf("  makespan %.2f ms | S0 busy %.2f (%d runs) | S1 busy %.2f (%d runs)"
           " | bubble ratio %.1f%%\n",
           makespan, busy[0], runs[0], busy[1], runs[1], bubble * 100.0);
}

/* ---- the two stage workers -------------------------------------------- */

static void *stage0_worker(void *arg) {
    Ring *r = arg;
    for (;;) {
        barrier_wait(&r->barrier);
        if (r->done) break;
        if (r->s0_req >= 0) {
            Request *q = r->req[r->s0_req];
            inflight_add(&r->in_flight, 1);
            inflight_peak(&r->max_in_flight, inflight_load(&r->in_flight));
            /* Rendezvous with the other stage so the overlap is witnessed
             * deterministically rather than depending on OS scheduling. */
            int expected = (r->s0_req >= 0) + (r->s1_req >= 0);
            while (inflight_load(&r->in_flight) < expected) sched_yield();
            double t0 = now_ms();
            int rc = run_stage(q->s0, 1, q->pos, &q->token, 1, q->emb, q->inter,
                               r->row_bytes, q->error, sizeof(q->error));
            trace_record(r, 0, r->s0_req, t0, now_ms());
            if (rc != 0) {
                r->failed = 1;
            } else {
                q->inter_token = q->token;
                q->inter_pos = q->pos;
                q->pos++;
                q->inter_pending = 1;
            }
            inflight_add(&r->in_flight, -1);
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
            Request *q = r->req[r->s1_req];
            inflight_add(&r->in_flight, 1);
            inflight_peak(&r->max_in_flight, inflight_load(&r->in_flight));
            int expected = (r->s0_req >= 0) + (r->s1_req >= 0);
            while (inflight_load(&r->in_flight) < expected) sched_yield();
            double t0 = now_ms();
            int rc = run_stage(q->s1, 1, q->inter_pos, &q->inter_token, 1,
                               q->inter, q->out, r->row_bytes,
                               q->error, sizeof(q->error));
            trace_record(r, 1, r->s1_req, t0, now_ms());
            if (rc != 0) {
                r->failed = 1;
            }
            inflight_add(&r->in_flight, -1);
        }
        barrier_wait(&r->barrier);
    }
    return NULL;
}

/* ---- prefill + lock-step ring ----------------------------------------- */

static int prefill(Request *q, ColiEdgeEngine *edge, size_t row_bytes,
                   const int32_t *prompt, int prompt_count,
                   char *error, size_t error_size) {
    size_t cap = (size_t)prompt_count * row_bytes;
    float *input = malloc(cap);
    float *inter = malloc(cap);
    float *out = malloc(cap);
    if (!input || !inter || !out) {
        free(out); free(inter); free(input);
        snprintf(error, error_size, "out of memory for prefill");
        return -1;
    }
    if (embed_tokens(edge, prompt, (size_t)prompt_count, input, row_bytes,
                     error, error_size) != 0)
        goto fail;
    if (run_stage(q->s0, (uint32_t)prompt_count, 0, prompt,
                  (size_t)prompt_count, input, inter, row_bytes,
                  error, error_size) != 0)
        goto fail;
    if (run_stage(q->s1, (uint32_t)prompt_count, 0, prompt,
                  (size_t)prompt_count, inter, out, row_bytes,
                  error, error_size) != 0)
        goto fail;
    if (select_token(edge,
                     out + (size_t)(prompt_count - 1) *
                               (row_bytes / sizeof(float)),
                     row_bytes, &q->token, error, error_size) != 0)
        goto fail;
    q->gen[0] = q->token;
    q->n_gen = 1;
    if (embed_tokens(edge, &q->token, 1, q->emb, row_bytes,
                     error, error_size) != 0)
        goto fail;
    q->pos = (uint64_t)prompt_count;
    q->inter_pending = 0;
    free(out); free(inter); free(input);
    return 0;
fail:
    free(out); free(inter); free(input);
    return -1;
}

static int pipelined_decode(Ring *r, int max_new,
                            char *error, size_t error_size) {
    pthread_t t0, t1;
    if (pthread_create(&t0, NULL, stage0_worker, r) != 0 ||
        pthread_create(&t1, NULL, stage1_worker, r) != 0) {
        snprintf(error, error_size, "cannot start pipeline workers");
        return -1;
    }

    int tick = 0;
    while ((r->req[0]->n_gen < max_new || r->req[1]->n_gen < max_new) &&
           !r->failed) {
        int even = (tick % 2 == 0);
        int r0 = even ? 0 : 1;
        int r1 = even ? 1 : 0;
        r->s0_req = (r->req[r0]->n_gen < max_new) ? r0 : -1;
        r->s1_req = r->req[r1]->inter_pending ? r1 : -1;
        barrier_wait(&r->barrier); /* release the two stages */
        barrier_wait(&r->barrier); /* wait for both stages to finish */
        if (r->s1_req >= 0 && !r->failed) {
            Request *q = r->req[r->s1_req];
            int32_t token = 0;
            double dt0 = now_ms();
            if (select_token(r->edge, q->out, r->row_bytes, &token,
                             error, error_size) != 0) {
                r->failed = 1;
            } else if (q->n_gen < max_new) {
                q->gen[q->n_gen++] = token;
                q->token = token;
                if (embed_tokens(r->edge, &token, 1, q->emb, r->row_bytes,
                                 error, error_size) != 0)
                    r->failed = 1;
            }
            trace_record(r, -1, r->s1_req, dt0, now_ms());
            q->inter_pending = 0;
        }
        tick++;
    }

    r->done = 1;
    barrier_wait(&r->barrier); /* wake the workers so they exit */
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    if (r->failed) {
        for (int i = 0; i < 2; i++) {
            if (r->req[i]->error[0]) {
                snprintf(error, error_size, "%s", r->req[i]->error);
                return -1;
            }
        }
        snprintf(error, error_size, "pipeline worker failed");
        return -1;
    }
    return 0;
}

/* ---- main -------------------------------------------------------------- */

static int open_stage(const char *family, const char *model_dir,
                      uint32_t begin, uint32_t end, uint32_t context,
                      ColiSegmentEngine **engine, ColiSegmentSession **session,
                      char *error, size_t error_size) {
    ColiSegmentEngineOptions options = {
        .struct_size = sizeof(options), .model_dir = model_dir,
        .layer_begin = begin, .layer_end = end, .context_tokens = context,
        /* request the CUDA-backed range when this binary links it; the engine
         * falls back to the CPU range internally when it declines */
        .backend_mask = COLI_SEGMENT_CAP_CPU | COLI_SEGMENT_CAP_CUDA,
    };
    if (coli_segment_engine_open(family, &options, engine, error, error_size) != 0) {
        /* a CPU-only build has no CUDA backend: open the CPU range instead */
        options.backend_mask = COLI_SEGMENT_CAP_CPU;
        if (coli_segment_engine_open(family, &options, engine, error, error_size) != 0)
            return -1;
    }
    ColiSegmentSessionOptions session_options = {
        .struct_size = sizeof(session_options), .context_tokens = context,
    };
    if (coli_segment_session_create(*engine, &session_options, session,
                                    error, error_size) != 0)
        return -1;
    return 0;
}

static void init_request(Request *q, ColiSegmentSession *s0,
                         ColiSegmentSession *s1, size_t row_bytes,
                         int max_new) {
    memset(q, 0, sizeof(*q));
    q->s0 = s0;
    q->s1 = s1;
    q->emb = malloc(row_bytes);
    q->inter = malloc(row_bytes);
    q->out = malloc(row_bytes);
    q->gen = malloc((size_t)max_new * sizeof(*q->gen));
}

static void free_request(Request *q) {
    if (!q) return;
    free(q->gen); free(q->out); free(q->inter); free(q->emb);
    free(q);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr,
                "usage: %s FAMILY MODEL_DIR SPLIT MAX_NEW [REF_JSON]\n",
                argv[0]);
        return 2;
    }
    const char *family = argv[1], *model_dir = argv[2];
    int split = atoi(argv[3]);
    int max_new = atoi(argv[4]);
    const char *ref_path = (argc == 6 && strcmp(argv[5], "-")) ? argv[5] : NULL;
    if (split < 1 || max_new < 2) {
        fprintf(stderr, "SPLIT must be >= 1 and MAX_NEW >= 2\n");
        return 2;
    }
    char error[512] = {0};
    char *json_text = NULL, *arena = NULL;
    jval *root = NULL;
    int32_t *prompt_a = NULL, *prompt_b = NULL, *full = NULL;
    int count_a = 0, count_b = 0, full_count = 0;
    int32_t *ref_a = NULL, *ref_b = NULL;
    ColiEdgeEngine *edge = NULL;
    ColiSegmentEngine *full_engine = NULL, *s0_engine = NULL, *s1_engine = NULL;
    ColiSegmentSession *full_session[2] = {NULL, NULL};
    ColiSegmentSession *s0_session[2] = {NULL, NULL};
    ColiSegmentSession *s1_session[2] = {NULL, NULL};
    Request *a = NULL, *b = NULL;
    Ring ring;
    int status = 1;
    memset(&ring, 0, sizeof(ring));

    REQUIRE(coli_qwen36_segment_adapter_register() == 0,
            "cannot register the qwen36 segment adapter");
    REQUIRE(coli_qwen36_edge_adapter_register() == 0,
            "cannot register the qwen36 edge adapter");

    if (ref_path) {
        json_text = read_file(ref_path);
        REQUIRE(json_text != NULL, "cannot read reference JSON");
        root = json_parse(json_text, &arena);
        REQUIRE(root && root->t == J_OBJ, "invalid reference JSON");
        prompt_a = read_ids(root, "prompt_ids", &count_a);
        full = read_ids(root, "full_ids", &full_count);
        REQUIRE(prompt_a && full && full_count > count_a,
                "reference token arrays are missing");
        if (max_new > full_count - count_a) max_new = full_count - count_a;
        REQUIRE(max_new >= 2, "reference has fewer than two generated tokens");
    }

    ColiEdgeEngineOptions edge_options = {
        .struct_size = sizeof(edge_options), .model_dir = model_dir,
    };
    REQUIRE(coli_edge_engine_open(family, &edge_options, &edge,
                                  error, sizeof(error)) == 0, error);
    ColiEdgeCapabilities edge_cap = {.struct_size = sizeof(edge_cap)};
    REQUIRE(coli_edge_engine_capabilities(edge, &edge_cap,
                                          error, sizeof(error)) == 0, error);
    REQUIRE(edge_cap.flags & COLI_EDGE_CAP_TOKENIZE,
            "edge adapter does not expose tokenization");
    REQUIRE(edge_cap.flags & COLI_EDGE_CAP_GREEDY,
            "edge adapter does not expose the model head");
    REQUIRE((uint32_t)split < edge_cap.num_layers,
            "split must fall inside the layer range");

    /* Request A uses the oracle prompt (or a fixed one); request B always
     * uses a distinct fixed prompt so the two streams are independent. */
    if (!prompt_a) {
        REQUIRE(tokenize(edge, "The quick brown fox", 19, &prompt_a, &count_a,
                         error, sizeof(error)) == 0,
                "fixed prompt A tokenization failed");
    }
    REQUIRE(tokenize(edge, "Colibri overlaps two decodes", 28,
                     &prompt_b, &count_b, error, sizeof(error)) == 0,
            "fixed prompt B tokenization failed");

    uint32_t context = (uint32_t)((count_a > count_b ? count_a : count_b) +
                                  max_new + 2);
    REQUIRE(open_stage(family, model_dir, 0, edge_cap.num_layers, context,
                       &full_engine, &full_session[0],
                       error, sizeof(error)) == 0, error);
    REQUIRE(open_stage(family, model_dir, 0, edge_cap.num_layers, context,
                       &full_engine, &full_session[1],
                       error, sizeof(error)) == 0, error);
    REQUIRE(open_stage(family, model_dir, 0, (uint32_t)split, context,
                       &s0_engine, &s0_session[0],
                       error, sizeof(error)) == 0, error);
    REQUIRE(open_stage(family, model_dir, 0, (uint32_t)split, context,
                       &s0_engine, &s0_session[1],
                       error, sizeof(error)) == 0, error);
    REQUIRE(open_stage(family, model_dir, (uint32_t)split, edge_cap.num_layers,
                       context, &s1_engine, &s1_session[0],
                       error, sizeof(error)) == 0, error);
    REQUIRE(open_stage(family, model_dir, (uint32_t)split, edge_cap.num_layers,
                       context, &s1_engine, &s1_session[1],
                       error, sizeof(error)) == 0, error);

    ColiSegmentCapabilities s0_cap = {.struct_size = sizeof(s0_cap)};
    ColiSegmentCapabilities s1_cap = {.struct_size = sizeof(s1_cap)};
    REQUIRE(coli_segment_engine_capabilities(s0_engine, &s0_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(coli_segment_engine_capabilities(s1_engine, &s1_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(s0_cap.state_dtype == edge_cap.state_dtype &&
            s0_cap.state_width == edge_cap.state_width &&
            !strcmp(s0_cap.state_schema, edge_cap.state_schema),
            "stage-0 Segment/Edge activation identity differs");
    REQUIRE(s1_cap.state_dtype == edge_cap.state_dtype &&
            s1_cap.state_width == edge_cap.state_width &&
            !strcmp(s1_cap.state_schema, edge_cap.state_schema),
            "stage-1 Segment/Edge activation identity differs");

    size_t row_bytes = (size_t)edge_cap.state_width * sizeof(float);
    ref_a = malloc((size_t)max_new * sizeof(*ref_a));
    ref_b = malloc((size_t)max_new * sizeof(*ref_b));
    REQUIRE(ref_a && ref_b, "out of memory for reference token buffers");

    double t_serial = now_ms();
    REQUIRE(serial_decode(full_session[0], edge, row_bytes, prompt_a, count_a,
                          max_new, ref_a, error, sizeof(error)) == 0, error);
    REQUIRE(serial_decode(full_session[1], edge, row_bytes, prompt_b, count_b,
                          max_new, ref_b, error, sizeof(error)) == 0, error);
    t_serial = now_ms() - t_serial;

    a = malloc(sizeof(*a));
    b = malloc(sizeof(*b));
    REQUIRE(a && b, "out of memory for pipeline requests");
    init_request(a, s0_session[0], s1_session[0], row_bytes, max_new);
    init_request(b, s0_session[1], s1_session[1], row_bytes, max_new);
    REQUIRE(a->emb && a->inter && a->out && a->gen &&
            b->emb && b->inter && b->out && b->gen,
            "out of memory for pipeline activation buffers");

    ring.req[0] = a;
    ring.req[1] = b;
    ring.edge = edge;
    ring.row_bytes = row_bytes;
    ring.trace_cap = 8 * max_new + 32;
    ring.trace = calloc((size_t)ring.trace_cap, sizeof(*ring.trace));
    REQUIRE(ring.trace != NULL, "out of memory for timeline trace");
    pthread_mutex_init(&ring.trace_lock, NULL);
    barrier_init(&ring.barrier, 3);

    double t_pipe = now_ms();
    REQUIRE(prefill(a, edge, row_bytes, prompt_a, count_a,
                    error, sizeof(error)) == 0, error);
    REQUIRE(prefill(b, edge, row_bytes, prompt_b, count_b,
                    error, sizeof(error)) == 0, error);
    REQUIRE(pipelined_decode(&ring, max_new, error, sizeof(error)) == 0, error);
    t_pipe = now_ms() - t_pipe;

    for (int i = 0; i < max_new; i++) {
        REQUIRE(a->gen[i] == ref_a[i], "pipelined request A differs from serial");
        REQUIRE(b->gen[i] == ref_b[i], "pipelined request B differs from serial");
        if (full)
            REQUIRE(ref_a[i] == full[count_a + i],
                    "request A differs from the independent oracle");
    }
    REQUIRE(ring.max_in_flight == 2,
            "the two stages never overlapped (max_in_flight != 2)");

    trace_report(&ring);
    printf("%s edge -> 2-stage ring (2 requests x %d tokens) ok "
           "(split %d/%u%s): overlap peak %d, serial %.2f ms, "
           "pipelined %.2f ms\n",
           family, max_new, split, edge_cap.num_layers,
           full ? ", oracle checked" : "", ring.max_in_flight,
           t_serial, t_pipe);
    status = 0;

fail:
    if (status && error[0]) fprintf(stderr, "%s: %s\n", family, error);
    if (ring.barrier.nparties) { /* destroy only if initialized */
        pthread_mutex_destroy(&ring.barrier.mu);
        pthread_cond_destroy(&ring.barrier.cv);
    }
    if (ring.trace) {
        pthread_mutex_destroy(&ring.trace_lock);
        free(ring.trace);
    }
    free_request(b);
    free_request(a);
    free(ref_b); free(ref_a);
    for (int i = 0; i < 2; i++) {
        coli_segment_session_destroy(s1_session[i]);
        coli_segment_session_destroy(s0_session[i]);
        coli_segment_session_destroy(full_session[i]);
    }
    if (s1_engine) (void)coli_segment_engine_close(s1_engine, NULL, 0);
    if (s0_engine) (void)coli_segment_engine_close(s0_engine, NULL, 0);
    if (full_engine) (void)coli_segment_engine_close(full_engine, NULL, 0);
    coli_edge_engine_close(edge);
    free(full); free(prompt_b); free(prompt_a);
    json_free(root); free(arena); free(json_text);
    return status;
}
