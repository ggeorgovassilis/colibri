#ifndef COLIBRI_KV_FP16_H
#define COLIBRI_KV_FP16_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* FP16 (IEEE 754 binary16) KV-cache for the attention K/V latent rows
 * (COLI_KV16). The latent compressed row is both key and value in the absorbed
 * MLA attention, so the quantization error lands in the scores AND the context
 * -- the same regime as COLI_KV8, but at half the f32 footprint with NO
 * per-row scale (fp16 carries its own exponent per element). On F16C hosts
 * (Ivy Bridge and later) decode is a single _mm_cvtph_ps per 4 lanes; the
 * scalar tier is a correct round-to-nearest-even reference, never on the hot
 * path there. Unlike COLI_KV8 there is no per-row scale array, so the cache is
 * K16/V16 = 2 bytes/element and nothing else.
 *
 * The fused dot/accumulate below keep the fp16 row in memory (no f32 staging):
 * that is the whole point -- a staged f32 row would discard the bandwidth win.
 * The reduction order therefore differs from dot_f32_lanes, so COLI_KV16 is
 * NOT bit-exact against the f32 path; it is a speed/quality trade measured in
 * benchmarks/colibri, not an oracle path. */

/* fp16 -> f32, exact for every bit pattern (zero, normal, subnormal, inf, NaN). */
static inline float coli_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormal or zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

/* f32 -> fp16, round-to-nearest-even (the hardware _mm_cvtps_ph default).
 * NaN -> qNaN (0x7E00), overflow -> inf (0x7C00), subnormal -> rounded fp16
 * subnormal or zero. */
static inline uint16_t coli_f32_to_f16(float f) {
    union { float f; uint32_t u; } v; v.f = f;
    uint32_t u = v.u;
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    uint32_t absu = u & 0x7FFFFFFFu;
    if (absu >= 0x7F800000u)                      /* Inf or NaN */
        return (uint16_t)(sign | ((absu > 0x7F800000u) ? 0x7E00u : 0x7C00u));
    if (absu >= 0x38800000u) {                    /* |f| >= 2^-14: normal */
        uint32_t r = absu + 0x00000FFFu + ((absu >> 13) & 1u);   /* RNE */
        if (r >= 0x47800000u) return (uint16_t)(sign | 0x7C00u); /* overflow */
        return (uint16_t)(sign | ((r - 0x38000000u) >> 13));
    }
    if (absu == 0) return sign;                   /* +/-0 */
    /* subnormal: round |f| * 2^24 to nearest int (ties to even) */
    float a = fabsf(f);
    int k = (int)roundf(a * 16777216.0f);
    if (k >= 1024) return (uint16_t)(sign | 0x0400u);  /* smallest normal */
    if (k <= 0) return sign;                            /* underflow to 0 */
    return (uint16_t)(sign | (uint16_t)k);
}

#if defined(__F16C__)
#include <immintrin.h>
#endif

/* f32 -> fp16 whole row. F16C _mm_cvtps_ph (RNE) on F16C hosts, scalar
 * reference elsewhere. The encode site is off the hot path (one row per token
 * per kv head), so the scalar tier is not a performance concern. */
static inline void coli_f32_to_f16_bulk(const float *src, uint16_t *dst, int64_t n) {
#if defined(__F16C__)
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 f = _mm_loadu_ps(src + i);
        __m128i h = _mm_cvtps_ph(f, 0);            /* imm 0 = round-to-nearest-even */
        _mm_storel_epi64((__m128i *)(dst + i), h);
    }
    for (; i < n; i++) dst[i] = coli_f32_to_f16(src[i]);
#else
    for (int64_t i = 0; i < n; i++) dst[i] = coli_f32_to_f16(src[i]);
#endif
}

/* Fused score dot acc = SUM q[j] * dequant(k16[j]), never staging a f32 row.
 * F16C converts 4 lanes per _mm_cvtph_ps; the reduction is a 4-wide tree then
 * a scalar tail, fixed order (not dot_f32_lanes' 32 lanes -- see header note). */
static inline float coli_dot_fp16(const float *q, const uint16_t *k16, int I) {
#if defined(__F16C__)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= I; i += 4) {
        __m128i h = _mm_loadl_epi64((const __m128i *)(k16 + i));
        __m128 w = _mm_cvtph_ps(h);
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(q + i), w));
    }
    float r[4]; _mm_storeu_ps(r, acc);
    float s = (r[0] + r[1]) + (r[2] + r[3]);
    for (; i < I; i++) s += q[i] * coli_f16_to_f32(k16[i]);
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < I; i++) s += q[i] * coli_f16_to_f32(k16[i]);
    return s;
#endif
}

/* Fused context accumulate: cx[0..I) += a * dequant(v16[0..I)), fp16 row never
 * staged. Used by the V context pass of the attention. */
static inline void coli_f16_madd(float *cx, const uint16_t *v16, float a, int I) {
#if defined(__F16C__)
    __m128 va = _mm_set1_ps(a);
    int i = 0;
    for (; i + 4 <= I; i += 4) {
        __m128i h = _mm_loadl_epi64((const __m128i *)(v16 + i));
        __m128 w = _mm_cvtph_ps(h);
        __m128 c = _mm_loadu_ps(cx + i);
        c = _mm_add_ps(c, _mm_mul_ps(va, w));
        _mm_storeu_ps(cx + i, c);
    }
    for (; i < I; i++) cx[i] += a * coli_f16_to_f32(v16[i]);
#else
    for (int i = 0; i < I; i++) cx[i] += a * coli_f16_to_f32(v16[i]);
#endif
}

#endif /* COLIBRI_KV_FP16_H */
