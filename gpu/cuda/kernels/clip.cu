// Scales each row of x, `width` long, by `scale`, then each head of it,
// `dim` long, down to a length of `threshold` if it is longer: a warp per
// head, its lanes taking the head's elements between them. And each row
// of x, `width` long, scaled by its own factor, a thread an element.

extern "C" __global__ void __launch_bounds__(32) clip(float* x, const float* scale, ClipParams p) {
    const u32 lane = threadIdx.x;
    float* head = x + (size_t)blockIdx.x * p.dim;
    const float* by = scale + (blockIdx.x % (p.width / p.dim)) * p.dim;
    float square = 0.0f;
    for (u32 j = lane; j < p.dim; j += 32) {
        float v = head[j] * by[j];
        square += v * v;
    }
    square = warp_sum(square);
    float factor = fminf(p.threshold / (sqrtf(square) + 1e-6f), 1.0f);
    for (u32 j = lane; j < p.dim; j += 32) {
        head[j] = head[j] * by[j] * factor;
    }
}

extern "C" __global__ void scale_rows(float* x, const float* factors, RowsParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.len) {
        x[i] *= factors[i / p.width];
    }
}

// x = cap * tanh(x / cap), a thread an element: the soft cap of the
// logits, with `width` for how many there are and `threshold` for the cap.
extern "C" __global__ void softcap(float* x, ClipParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.width) {
        x[i] = p.threshold * tanhf(x[i] / p.threshold);
    }
}

// Several clips in one grid, each job's own activations, scale, widths
// and threshold: the heads of one job after another's, `first[j]` being
// the first block of job j, and a job past the last starting at the end.
#define CLIP_JOBS 4

struct ClipJobs {
    float* x[CLIP_JOBS];
    const float* scale[CLIP_JOBS];
    u32 first[CLIP_JOBS];
    u32 width[CLIP_JOBS];
    u32 dim[CLIP_JOBS];
    float threshold[CLIP_JOBS];
};

extern "C" __global__ void __launch_bounds__(32) clips(ClipJobs p) {
    u32 j = 0;
#pragma unroll
    for (u32 i = 1; i < CLIP_JOBS; i++) {
        j += blockIdx.x >= p.first[i];
    }
    const u32 lane = threadIdx.x;
    const u32 block = blockIdx.x - p.first[j];
    const u32 dim = p.dim[j];
    float* head = p.x[j] + (size_t)block * dim;
    const float* by = p.scale[j] + (block % (p.width[j] / dim)) * dim;
    float square = 0.0f;
    for (u32 i = lane; i < dim; i += 32) {
        float v = head[i] * by[i];
        square += v * v;
    }
    square = warp_sum(square);
    float factor = fminf(p.threshold[j] / (sqrtf(square) + 1e-6f), 1.0f);
    for (u32 i = lane; i < dim; i += 32) {
        head[i] = head[i] * by[i] * factor;
    }
}
