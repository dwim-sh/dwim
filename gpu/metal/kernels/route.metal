// Picks each token's experts: the k whose score is highest, the score of
// each being the sigmoid of its logit plus its bias, the lower number
// first of two the same. ids gets the experts' numbers, the highest score
// first, and weights what each one's output counts for: its sigmoid over
// the sum of the k, times `scale`. A thread per token.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint experts;
    uint k;
    float scale;
};

// Most experts there are, and most a token goes through.
constant uint EXPERTS = 256;
constant uint PICKED = 32;

kernel void route(
    device float* ids [[buffer(0)]],
    device float* weights [[buffer(1)]],
    const device float* logits [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint t [[thread_position_in_grid]])
{
    if (t >= p.n) {
        return;
    }
    float probs[EXPERTS];
    bool taken[EXPERTS];
    for (uint e = 0; e < p.experts; e++) {
        probs[e] = 1.0f / (1.0f + exp(-logits[t * p.experts + e]));
        taken[e] = false;
    }
    uint picked[PICKED];
    float sum = 0.0f;
    for (uint j = 0; j < p.k; j++) {
        uint best = 0;
        float most = -INFINITY;
        for (uint e = 0; e < p.experts; e++) {
            float score = probs[e] + bias[e];
            if (!taken[e] && score > most) {
                best = e;
                most = score;
            }
        }
        taken[best] = true;
        picked[j] = best;
        sum += probs[best];
    }
    for (uint j = 0; j < p.k; j++) {
        ids[t * p.k + j] = float(picked[j]);
        weights[t * p.k + j] = probs[picked[j]] / (sum + 1e-20f) * p.scale;
    }
}
