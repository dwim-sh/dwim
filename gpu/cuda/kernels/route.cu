// Picks each token's experts: the k whose score is highest, the score of
// each being the sigmoid of its logit plus its bias, the lower number
// first of two the same. ids gets the experts' numbers, the highest score
// first, and weights what each one's output counts for: its sigmoid over
// the sum of the k, times `scale`. A warp per token: its lanes take the
// experts between them, and each pick is the best of the lanes' best.

// Most experts there are, and so most a lane takes.
#define ROUTE_EXPERTS 256
#define ROUTE_PER_LANE (ROUTE_EXPERTS / 32)
// A score below any an expert has.
#define ROUTE_NONE (-3.0e38f)

extern "C" __global__ void __launch_bounds__(32) route(
    float* ids, float* weights, const float* logits, const float* bias, RouteParams p) {
    const u32 lane = threadIdx.x;
    const u32 t = blockIdx.x;
    // The lane's experts' sigmoids and scores: an expert picked, or one
    // there is not, scores below any.
    float probs[ROUTE_PER_LANE];
    float scores[ROUTE_PER_LANE];
#pragma unroll
    for (u32 i = 0; i < ROUTE_PER_LANE; i++) {
        u32 e = lane + i * 32;
        probs[i] = 0.0f;
        scores[i] = ROUTE_NONE;
        if (e < p.experts) {
            probs[i] = sigmoid(logits[(size_t)t * p.experts + e]);
            scores[i] = probs[i] + bias[e];
        }
    }
    float sum = 0.0f;
    float picked = 0.0f;
    for (u32 j = 0; j < p.k; j++) {
        float most = ROUTE_NONE;
        u32 best = 0xffffffffu;
#pragma unroll
        for (u32 i = 0; i < ROUTE_PER_LANE; i++) {
            if (scores[i] > most) {
                most = scores[i];
                best = lane + i * 32;
            }
        }
        // The best across the warp: the higher score, and the lower
        // number of two the same.
#pragma unroll
        for (u32 s = 16; s > 0; s >>= 1) {
            float other = __shfl_xor_sync(FULL, most, s);
            u32 theirs = __shfl_xor_sync(FULL, best, s);
            if (other > most || (other == most && theirs < best)) {
                most = other;
                best = theirs;
            }
        }
        float prob = 0.0f;
        if (best % 32 == lane) {
            prob = probs[best / 32];
            scores[best / 32] = ROUTE_NONE;
        }
        prob = warp_sum(prob);
        sum += prob;
        // Lane j keeps pick j, to write when the sum is known.
        if (lane == j) {
            picked = prob;
            ids[(size_t)t * p.k + j] = (float)best;
        }
    }
    if (lane < p.k) {
        weights[(size_t)t * p.k + lane] = picked / (sum + 1e-20f) * p.scale;
    }
}
