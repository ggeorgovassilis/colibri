#include "../edge_adapters.h"
#include "../edge_runtime.h"
#include "../json.h"
#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * M1 pipeline driver: drive a greedy decode through a 2-stage Segment chain
 * (edge -> stage[0, split) -> stage[split, n) -> edge) and prove it yields the
 * same token stream as the single full-range Segment chain. When a reference
 * JSON is supplied the decode is additionally checked against the independent
 * oracle recorded there.
 *
 * This is the in-process pipeline driver that a later distributed/multi-device
 * driver will reimplement over the wire. Functionality, correctness and
 * observability come first; throughput is not measured here.
 *
 * usage: test_pipeline_real FAMILY MODEL_DIR SPLIT MAX_NEW [REF_JSON]
 *   REF_JSON may be "-" to skip the oracle and drive a fixed text prompt.
 */

#define REQUIRE(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "%s: %s\n", family, message); goto fail; } \
} while (0)

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

/* One full-range engine or a two-stage [0, split) + [split, n) chain. */
typedef struct {
    const char *name;
    ColiSegmentEngine *engines[2];
    ColiSegmentSession *sessions[2];
    int stages;
} Chain;

static int chain_run(const Chain *c, uint32_t rows, uint64_t position,
                     const int32_t *token_ids, size_t token_count,
                     const float *input, float *output, float *scratch,
                     size_t row_bytes, char *error, size_t error_size) {
    const float *in = input;
    float *out = (c->stages == 2) ? scratch : output;
    for (int stage = 0; stage < c->stages; stage++) {
        ColiSegmentRunRequest run = {
            .struct_size = sizeof(run), .rows = rows,
            .position = position,
            .token_ids = token_ids, .token_count = token_count,
            .input = in, .input_bytes = (size_t)rows * row_bytes,
            .output = out, .output_bytes = (size_t)rows * row_bytes,
        };
        if (coli_segment_run(c->sessions[stage], &run, error, error_size) != 0)
            return -1;
        if (c->stages == 2) { in = scratch; out = output; }
    }
    return 0;
}

/* Greedy decode through `c`, returning `max_new` generated tokens. The first
 * row of every scratch buffer is reused for the single-token decode phase. */
static int decode(const Chain *c, ColiEdgeEngine *edge, size_t row_bytes,
                  const int32_t *prompt, int prompt_count, int max_new,
                  int32_t *tokens, char *error, size_t error_size) {
    size_t rows_cap = (size_t)prompt_count;
    float *input = malloc(rows_cap * row_bytes);
    float *output = malloc(rows_cap * row_bytes);
    float *scratch = malloc(rows_cap * row_bytes);
    if (!input || !output || !scratch) {
        free(scratch); free(output); free(input);
        snprintf(error, error_size, "out of memory for decode activations");
        return -1;
    }

    ColiEdgeEmbedRequest embed = {
        .struct_size = sizeof(embed), .rows = (uint32_t)prompt_count,
        .token_ids = prompt, .token_count = (size_t)prompt_count,
        .output = input, .output_bytes = (size_t)prompt_count * row_bytes,
    };
    if (coli_edge_embed(edge, &embed, error, error_size) != 0) goto decode_fail;
    if (chain_run(c, (uint32_t)prompt_count, 0, prompt, (size_t)prompt_count,
                  input, output, scratch, row_bytes, error, error_size) != 0)
        goto decode_fail;

    float score = 0.0f;
    ColiEdgeSelectRequest select = {
        .struct_size = sizeof(select), .rows = 1,
        .input = output + (size_t)(prompt_count - 1) * (row_bytes / sizeof(float)),
        .input_bytes = row_bytes, .token_ids = &tokens[0],
        .token_capacity = 1, .scores = &score, .score_capacity = 1,
    };
    if (coli_edge_select(edge, &select, error, error_size) != 0) goto decode_fail;

    for (int generated = 1; generated < max_new; generated++) {
        int32_t token = tokens[generated - 1];
        embed.rows = 1; embed.token_ids = &token; embed.token_count = 1;
        embed.output = input; embed.output_bytes = row_bytes;
        if (coli_edge_embed(edge, &embed, error, error_size) != 0) goto decode_fail;
        if (chain_run(c, 1, (uint64_t)prompt_count + (uint64_t)generated - 1u,
                      &token, 1, input, output, scratch, row_bytes,
                      error, error_size) != 0)
            goto decode_fail;
        select.input = output;
        select.token_ids = &tokens[generated];
        if (coli_edge_select(edge, &select, error, error_size) != 0) goto decode_fail;
    }

    free(scratch); free(output); free(input);
    return 0;

decode_fail:
    free(scratch); free(output); free(input);
    return -1;
}

static int open_stage(const char *family, const char *model_dir,
                      uint32_t begin, uint32_t end, uint32_t context,
                      ColiSegmentEngine **engine, ColiSegmentSession **session,
                      char *error, size_t error_size) {
    ColiSegmentEngineOptions options = {
        .struct_size = sizeof(options), .model_dir = model_dir,
        .layer_begin = begin, .layer_end = end, .context_tokens = context,
    };
    if (coli_segment_engine_open(family, &options, engine, error, error_size) != 0)
        return -1;
    ColiSegmentSessionOptions session_options = {
        .struct_size = sizeof(session_options), .context_tokens = context,
    };
    if (coli_segment_session_create(*engine, &session_options, session,
                                    error, error_size) != 0)
        return -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s FAMILY MODEL_DIR SPLIT MAX_NEW [REF_JSON]\n",
                argv[0]);
        return 2;
    }
    const char *family = argv[1], *model_dir = argv[2];
    int split = atoi(argv[3]);
    int max_new = atoi(argv[4]);
    const char *ref_path = (argc == 6 && strcmp(argv[5], "-")) ? argv[5] : NULL;
    if (split < 1 || max_new < 1) return 2;
    char error[512] = {0};
    char *json_text = NULL, *arena = NULL;
    jval *root = NULL;
    int32_t *prompt = NULL, *full = NULL;
    int prompt_count = 0, full_count = 0;
    int32_t *tokens_full = NULL, *tokens_pipe = NULL;
    ColiEdgeEngine *edge = NULL;
    ColiSegmentEngine *full_engine = NULL, *s0_engine = NULL, *s1_engine = NULL;
    ColiSegmentSession *full_session = NULL, *s0_session = NULL, *s1_session = NULL;
    int status = 1;

    REQUIRE(coli_qwen36_segment_adapter_register() == 0,
            "cannot register the qwen36 segment adapter");
    REQUIRE(coli_qwen36_edge_adapter_register() == 0,
            "cannot register the qwen36 edge adapter");

    if (ref_path) {
        json_text = read_file(ref_path);
        REQUIRE(json_text != NULL, "cannot read reference JSON");
        root = json_parse(json_text, &arena);
        REQUIRE(root && root->t == J_OBJ, "invalid reference JSON");
        prompt = read_ids(root, "prompt_ids", &prompt_count);
        full = read_ids(root, "full_ids", &full_count);
        REQUIRE(prompt && full && full_count > prompt_count,
                "reference token arrays are missing");
        if (max_new > full_count - prompt_count) max_new = full_count - prompt_count;
        REQUIRE(max_new >= 1, "reference has no generated tokens");
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

    if (!prompt) {
        static const char probe[] = "The quick brown fox";
        size_t probe_count = 0;
        REQUIRE(coli_edge_tokenize(edge, probe, sizeof(probe) - 1u,
                                   NULL, 0, &probe_count,
                                   error, sizeof(error)) == 0 && probe_count,
                "fixed-prompt tokenization sizing failed");
        prompt = malloc(probe_count * sizeof(*prompt));
        REQUIRE(prompt != NULL, "out of memory for fixed prompt");
        REQUIRE(coli_edge_tokenize(edge, probe, sizeof(probe) - 1u,
                                   prompt, probe_count, &probe_count,
                                   error, sizeof(error)) == 0,
                "fixed-prompt tokenization failed");
        prompt_count = (int)probe_count;
    }

    uint32_t context = (uint32_t)(prompt_count + max_new + 2);
    REQUIRE(open_stage(family, model_dir, 0, edge_cap.num_layers, context,
                       &full_engine, &full_session, error, sizeof(error)) == 0,
            error);
    REQUIRE(open_stage(family, model_dir, 0, (uint32_t)split, context,
                       &s0_engine, &s0_session, error, sizeof(error)) == 0,
            error);
    REQUIRE(open_stage(family, model_dir, (uint32_t)split, edge_cap.num_layers,
                       context, &s1_engine, &s1_session,
                       error, sizeof(error)) == 0, error);

    ColiSegmentCapabilities full_cap = {.struct_size = sizeof(full_cap)};
    ColiSegmentCapabilities s0_cap = {.struct_size = sizeof(s0_cap)};
    ColiSegmentCapabilities s1_cap = {.struct_size = sizeof(s1_cap)};
    REQUIRE(coli_segment_engine_capabilities(full_engine, &full_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(coli_segment_engine_capabilities(s0_engine, &s0_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(coli_segment_engine_capabilities(s1_engine, &s1_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(full_cap.state_dtype == edge_cap.state_dtype &&
            full_cap.state_width == edge_cap.state_width &&
            !strcmp(full_cap.state_schema, edge_cap.state_schema) &&
            !strcmp(full_cap.numeric_class, edge_cap.numeric_class),
            "full-range Segment/Edge activation identity differs");
    REQUIRE(s0_cap.state_dtype == edge_cap.state_dtype &&
            s0_cap.state_width == edge_cap.state_width &&
            !strcmp(s0_cap.state_schema, edge_cap.state_schema) &&
            !strcmp(s0_cap.numeric_class, edge_cap.numeric_class),
            "stage-0 Segment/Edge activation identity differs");
    REQUIRE(s1_cap.state_dtype == edge_cap.state_dtype &&
            s1_cap.state_width == edge_cap.state_width &&
            !strcmp(s1_cap.state_schema, edge_cap.state_schema) &&
            !strcmp(s1_cap.numeric_class, edge_cap.numeric_class),
            "stage-1 Segment/Edge activation identity differs");

    size_t row_bytes = (size_t)edge_cap.state_width * sizeof(float);
    Chain full_chain = {"full", {full_engine, NULL}, {full_session, NULL}, 1};
    Chain pipe_chain = {"2-stage", {s0_engine, s1_engine},
                        {s0_session, s1_session}, 2};
    tokens_full = malloc((size_t)max_new * sizeof(*tokens_full));
    tokens_pipe = malloc((size_t)max_new * sizeof(*tokens_pipe));
    REQUIRE(tokens_full && tokens_pipe, "out of memory for token buffers");

    REQUIRE(decode(&full_chain, edge, row_bytes, prompt, prompt_count, max_new,
                   tokens_full, error, sizeof(error)) == 0, error);
    REQUIRE(decode(&pipe_chain, edge, row_bytes, prompt, prompt_count, max_new,
                   tokens_pipe, error, sizeof(error)) == 0, error);

    for (int i = 0; i < max_new; i++) {
        REQUIRE(tokens_pipe[i] == tokens_full[i],
                "2-stage token differs from the full-range chain");
        if (full)
            REQUIRE(tokens_full[i] == full[prompt_count + i],
                    "full-range token differs from the independent oracle");
    }
    printf("%s edge -> %d-stage Segment -> edge: %d tokens ok (split %d/%u%s)\n",
           family, pipe_chain.stages, max_new, split, edge_cap.num_layers,
           full ? ", oracle checked" : "");
    status = 0;

fail:
    if (status && error[0]) fprintf(stderr, "%s: %s\n", family, error);
    free(tokens_pipe); free(tokens_full);
    coli_segment_session_destroy(s1_session);
    coli_segment_session_destroy(s0_session);
    coli_segment_session_destroy(full_session);
    if (s1_engine) (void)coli_segment_engine_close(s1_engine, NULL, 0);
    if (s0_engine) (void)coli_segment_engine_close(s0_engine, NULL, 0);
    if (full_engine) (void)coli_segment_engine_close(full_engine, NULL, 0);
    coli_edge_engine_close(edge);
    free(full); free(prompt);
    json_free(root); free(arena); free(json_text);
    return status;
}
