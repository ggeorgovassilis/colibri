/*
 * M2 synthetic overlap proof: a deterministic, model-free micro-benchmark that
 * demonstrates the 2-stage pipeline's temporal overlap. Two synthetic requests
 * each push STEPS tokens through two "stages" of fixed cost. Run serially the
 * total work is 4*STEPS stage-times; with the lock-step ring the two stages run
 * on separate threads and overlap, so the wall-clock must drop well below the
 * serial sum. This is the same scheduling discipline as test_pipeline_overlap.c
 * but with a synthetic work function instead of the Segment engine, so the
 * result is reproducible and CI-safe (no model, no GPU, no fixture).
 *
 * Observability: per-stage busy time and the pipeline bubble ratio are computed
 * for both the serial and pipelined runs. The serialised run reports ~50%
 * bubble (the two stages never overlap); the pipelined run reports only the
 * fill/drain ramp as bubble. The assertions pin both effects down.
 *
 * usage: test_pipeline_synthetic [WORK_US] [STEPS]
 *   WORK_US defaults to 20000 (20 ms per stage invocation).
 *   STEPS defaults to 8 (tokens per request).
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void work(int us) {
    struct timespec ts = {us / 1000000, (long)(us % 1000000) * 1000};
    while (nanosleep(&ts, &ts) != 0) { /* retry on EINTR */ }
}

/* ---- reusable barrier (POSIX) ----------------------------------------- */

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

/* ---- per-request + ring state ----------------------------------------- */

typedef struct {
    int remaining;  /* stage-0 jobs still to do (tokens yet to enter stage 0) */
    int inflight;   /* a token is waiting to drain through stage 1 */
} SynReq;

typedef struct {
    SynReq *req[2];
    int work_us;
    barrier_t barrier;  /* 3 parties: driver + 2 stage workers */
    int s0_req, s1_req; /* request index each stage runs, or -1 */
    int done;
    /* observability */
    double busy[2];
    int runs[2];
} Ring;

/* ---- the two stage workers -------------------------------------------- */

static void *stage0_worker(void *arg) {
    Ring *r = arg;
    for (;;) {
        barrier_wait(&r->barrier);
        if (r->done) break;
        if (r->s0_req >= 0) {
            double t0 = now_ms();
            work(r->work_us);
            r->busy[0] += now_ms() - t0;
            r->runs[0]++;
            r->req[r->s0_req]->remaining--;
            r->req[r->s0_req]->inflight = 1;
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
            double t0 = now_ms();
            work(r->work_us);
            r->busy[1] += now_ms() - t0;
            r->runs[1]++;
            r->req[r->s1_req]->inflight = 0;
        }
        barrier_wait(&r->barrier);
    }
    return NULL;
}

/* ---- serial reference: one stage at a time ----------------------------- */

static double run_serial(int steps, int work_us, double *busy_total) {
    double t0 = now_ms();
    for (int rq = 0; rq < 2; rq++)
        for (int s = 0; s < steps; s++) {
            work(work_us); /* stage 0 */
            work(work_us); /* stage 1 */
        }
    double t1 = now_ms();
    *busy_total = (double)(4 * steps * work_us) / 1000.0;
    return t1 - t0;
}

/* ---- pipelined lock-step ring ----------------------------------------- */

static double run_pipelined(Ring *r, int steps) {
    pthread_t t0, t1;
    if (pthread_create(&t0, NULL, stage0_worker, r) != 0 ||
        pthread_create(&t1, NULL, stage1_worker, r) != 0) {
        fprintf(stderr, "cannot start pipeline workers\n");
        return -1.0;
    }
    int tick = 0;
    double start = now_ms();
    while (r->req[0]->remaining > 0 || r->req[0]->inflight ||
           r->req[1]->remaining > 0 || r->req[1]->inflight) {
        int even = (tick % 2 == 0);
        int r0 = even ? 0 : 1, r1 = even ? 1 : 0;
        r->s0_req = r->req[r0]->remaining > 0 ? r0 : -1;
        r->s1_req = r->req[r1]->inflight ? r1 : -1;
        barrier_wait(&r->barrier); /* release the two stages */
        barrier_wait(&r->barrier); /* wait for both stages to finish */
        tick++;
    }
    double end = now_ms();
    r->done = 1;
    barrier_wait(&r->barrier); /* wake the workers so they exit */
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);
    (void)steps;
    return end - start;
}

/* ---- main -------------------------------------------------------------- */

int main(int argc, char **argv) {
    int work_us = 20000;
    int steps = 8;
    if (argc > 1) work_us = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (work_us < 1000 || steps < 2) {
        fprintf(stderr, "WORK_US must be >= 1000 and STEPS >= 2\n");
        return 2;
    }

    double serial_busy = 0.0;
    double serial_ms = run_serial(steps, work_us, &serial_busy);
    double serial_bubble =
        1.0 - serial_busy / (2.0 * serial_ms);

    SynReq a = {steps, 0}, b = {steps, 0};
    Ring r;
    memset(&r, 0, sizeof(r));
    r.req[0] = &a;
    r.req[1] = &b;
    r.work_us = work_us;
    barrier_init(&r.barrier, 3);

    double pipe_ms = run_pipelined(&r, steps);
    pthread_mutex_destroy(&r.barrier.mu);
    pthread_cond_destroy(&r.barrier.cv);

    double pipe_busy = r.busy[0] + r.busy[1];
    double pipe_bubble = pipe_ms > 0.0
        ? 1.0 - pipe_busy / (2.0 * pipe_ms) : 1.0;
    double speedup = pipe_ms > 0.0 ? serial_ms / pipe_ms : 0.0;

    printf("synthetic 2-stage x 2-request overlap (%d steps x %d us work):\n",
           steps, work_us);
    printf("  serial     %.2f ms | busy %.2f ms | bubble %.1f%%\n",
           serial_ms, serial_busy, serial_bubble * 100.0);
    printf("  pipelined  %.2f ms | busy %.2f ms | bubble %.1f%% | %.2fx\n",
           pipe_ms, pipe_busy, pipe_bubble * 100.0, speedup);

    if (pipe_ms < 0.0) return 1;
    if (!(pipe_ms < serial_ms * 0.85)) {
        fprintf(stderr,
                "FAIL: pipelined (%.2f ms) did not beat serial by >= 15%% "
                "(serial %.2f ms)\n",
                pipe_ms, serial_ms);
        return 1;
    }
    if (!(pipe_bubble < serial_bubble)) {
        fprintf(stderr,
                "FAIL: pipelined bubble (%.1f%%) did not improve over serial "
                "(%.1f%%)\n",
                pipe_bubble * 100.0, serial_bubble * 100.0);
        return 1;
    }
    printf("ok: wall-clock < serial sum and bubble ratio improved\n");
    return 0;
}
