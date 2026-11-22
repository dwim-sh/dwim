// Picks each token's experts: the k whose score is highest, the score of
// each being the sigmoid of its logit plus its bias, the lower number
// first of two the same. ids gets the experts' numbers, the highest score
// first, and weights what each one's output counts for: its sigmoid over
// the sum of the k, times `scale`. A thread per token, which looks over
// the experts once for each it picks: the one that comes next after the
// last picked.

struct Params {
    n: u32,
    experts: u32,
    k: u32,
    scale: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> ids: array<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> logits: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

// Most experts a token goes through.
const PICKED: u32 = 32u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= p.n {
        return;
    }
    var picked: array<u32, PICKED>;
    var probs: array<f32, PICKED>;
    var sum = 0.0;
    var last = 0u;
    var last_score = 0.0;
    for (var j = 0u; j < p.k; j++) {
        var best = 0u;
        var most = 0.0;
        var prob = 0.0;
        var found = false;
        for (var e = 0u; e < p.experts; e++) {
            let sigmoid = 1.0 / (1.0 + exp(-logits[t * p.experts + e]));
            let score = sigmoid + bias[e];
            let after = j == 0u || score < last_score || (score == last_score && e > last);
            if after && (!found || score > most) {
                best = e;
                most = score;
                prob = sigmoid;
                found = true;
            }
        }
        picked[j] = best;
        probs[j] = prob;
        sum += prob;
        last = best;
        last_score = most;
    }
    for (var j = 0u; j < p.k; j++) {
        ids[t * p.k + j] = f32(picked[j]);
        weights[t * p.k + j] = probs[j] / (sum + 1e-20) * p.scale;
    }
}
