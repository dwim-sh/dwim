// Picks each token's experts: the k whose score is highest, the score of
// each being the sigmoid of its logit plus its bias, the lower number
// first of two the same. ids gets the experts' numbers, the highest score
// first, and weights what each one's output counts for: its sigmoid over
// the sum of the k, times `scale`. One SIMD group per token: its lanes
// hold the experts' probabilities and scores in registers and pick the
// best across the group with shuffles.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint experts;
    uint k;
    float scale;
};

// Most experts there are, and most a lane takes.
constant uint EXPERTS = 256;
constant uint PER_LANE = EXPERTS / 32;
// A score below any an expert has.
constant float NONE = -3.0e38f;

kernel void route(
    device float* ids [[buffer(0)]],
    device float* weights [[buffer(1)]],
    const device float* logits [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint t [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (t >= p.n) {
        return;
    }
    float probs[PER_LANE];
    float scores[PER_LANE];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < PER_LANE; i++) {
        uint e = lane + i * 32;
        probs[i] = 0.0f;
        scores[i] = NONE;
        if (e < p.experts) {
            probs[i] = 1.0f / (1.0f + exp(-logits[t * p.experts + e]));
            scores[i] = probs[i] + bias[e];
        }
    }
    float sum = 0.0f;
    float picked = 0.0f;
    for (uint j = 0; j < p.k; j++) {
        float most = NONE;
        uint best = 0xffffffffu;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < PER_LANE; i++) {
            if (scores[i] > most) {
                most = scores[i];
                best = lane + i * 32;
            }
        }
        // The best across the SIMD group: the higher score, and the lower
        // number of two the same.
        #pragma clang loop unroll(full)
        for (ushort s = 16; s > 0; s >>= 1) {
            float other = simd_shuffle_xor(most, s);
            uint theirs = simd_shuffle_xor(best, s);
            if (other > most || (other == most && theirs < best)) {
                most = other;
                best = theirs;
            }
        }
        float prob = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < PER_LANE; i++) {
            if (lane + i * 32 == best) {
                prob = probs[i];
                scores[i] = NONE;
            }
        }
        prob = simd_sum(prob);
        sum += prob;
        if (lane == j) {
            picked = prob;
            ids[t * p.k + j] = float(best);
        }
    }
    if (lane < p.k) {
        weights[t * p.k + lane] = picked / (sum + 1e-20f) * p.scale;
    }
}
