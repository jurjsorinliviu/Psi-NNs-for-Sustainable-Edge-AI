"""
export_c.py -- emit a trained model as self-contained C for a Cortex-M target.

Three storage formats, matching the three claims the paper makes about memory:

  float32    weights as fp32                                    -> 4 B / param
  int8       symmetric per-tensor INT8, dequantized on the fly  -> 1 B / param
  clustered  the Psi-NN compressed form (Eq. 16, vec(W) = R mu):
             K centroids in flash plus ONE SIGNED BYTE per weight holding the cluster
             index and its sign. Weights are dereferenced on the fly, so nothing is
             expanded into RAM.  ->  4*K + N bytes

The clustered format is what makes the paper's "M_weights = sum_l K_l * S_param"
claim concrete on a real device: the centroid table is tiny, but the index table is
not free, and this exporter counts both honestly.

The kernel is plain C (no CMSIS), so the same source builds for Cortex-M4 and M7 and
its instruction count can be compared against the operation model of Eqs. (18)-(20).
"""

import numpy as np
import torch


def _fa(name, v, per_line=8):
    b = [", ".join(f"{x:.8e}f" for x in v[i:i + per_line]) for i in range(0, len(v), per_line)]
    return f"const float {name}[{len(v)}] = {{\n    " + ",\n    ".join(b) + "\n};\n"


def _ia(name, v, ctype="int8_t", per_line=16):
    b = [", ".join(str(int(x)) for x in v[i:i + per_line]) for i in range(0, len(v), per_line)]
    return f"const {ctype} {name}[{len(v)}] = {{\n    " + ",\n    ".join(b) + "\n};\n"


def _quantize_int8(v):
    s = float(np.abs(v).max()) / 127.0
    if s <= 0:
        s = 1.0
    return np.clip(np.round(v / s), -127, 127).astype(np.int8), s


def _cluster(v, eps):
    """Per-tensor HAC on |w| at threshold eps -> (centroids, signed 1-based index)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    a = np.abs(v)
    if v.size == 1:
        lab = np.array([1])
    else:
        lab = fcluster(linkage(a.reshape(-1, 1), method="average"), t=eps,
                       criterion="distance")
    uniq = np.unique(lab)
    if len(uniq) > 127:
        raise ValueError(f"{len(uniq)} clusters exceeds the 127 encodable in a signed byte")
    mu = np.array([a[lab == u].mean() for u in uniq], dtype=np.float32)
    remap = {u: i + 1 for i, u in enumerate(uniq)}          # 1-based so sign is free
    idx = np.array([remap[l] * (1 if x >= 0 else -1) for l, x in zip(lab, v)], dtype=np.int8)
    return mu, idx


class _Emitter:
    """Emits one weight tensor in the chosen format and gives C code to read element k."""

    def __init__(self, fmt, eps=None):
        self.fmt = fmt
        self.eps = eps
        self.decls = []
        self.weight_bytes = 0
        self.n_clusters = 0

    def add(self, name, v):
        v = np.asarray(v, dtype=np.float32).ravel()
        if self.fmt == "float32":
            self.decls.append(_fa(name, v))
            self.weight_bytes += v.size * 4
            return lambda k: f"{name}[{k}]"
        if self.fmt == "int8":
            q, s = _quantize_int8(v)
            self.decls.append(_ia(name, q))
            self.decls.append(f"const float {name}_s = {s:.8e}f;\n")
            self.weight_bytes += v.size + 4
            return lambda k: f"({name}[{k}] * {name}_s)"
        if self.fmt == "clustered":
            mu, idx = _cluster(v, self.eps)
            self.decls.append(_fa(f"{name}_mu", mu))
            self.decls.append(_ia(f"{name}_i", idx))
            self.weight_bytes += mu.size * 4 + idx.size
            self.n_clusters += mu.size
            return lambda k: (f"(({name}_i[{k}] < 0) ? -{name}_mu[-{name}_i[{k}] - 1]"
                              f" : {name}_mu[{name}_i[{k}] - 1])")
        raise ValueError(self.fmt)

    def add_bias(self, name, v):
        """Biases are always fp32: they are few, and quantizing them buys nothing."""
        v = np.asarray(v, dtype=np.float32).ravel()
        self.decls.append(_fa(name, v))
        self.weight_bytes += v.size * 4
        return lambda k: f"{name}[{k}]"

    def add_precomputed_cluster(self, name, mu, signed_idx):
        """Emit an already-clustered tensor (centroids + signed 1-based index).

        Used to ship the DISTILLED, centroid-retrained model of Ref. [23] rather than a
        post-hoc clustering of the trained weights -- the two have very different
        accuracy (exp4), and only the former backs the paper's memory claim.
        """
        mu = np.asarray(mu, dtype=np.float32).ravel()
        signed_idx = np.asarray(signed_idx, dtype=np.int8).ravel()
        if mu.size > 127:
            raise ValueError(f"{mu.size} centroids exceed the 127 encodable in a signed byte")
        self.decls.append(_fa(f"{name}_mu", mu))
        self.decls.append(_ia(f"{name}_i", signed_idx))
        self.weight_bytes += mu.size * 4 + signed_idx.size
        self.n_clusters += mu.size
        return lambda k: (f"(({name}_i[{k}] < 0) ? -{name}_mu[-{name}_i[{k}] - 1]"
                          f" : {name}_mu[{name}_i[{k}] - 1])")


def cluster_tensor_from(cm, pname, transform):
    """Pull (centroids, signed 1-based index) for one parameter out of a ClusteredModel,
    applying the same flattening `transform` the C kernel expects for that tensor."""
    key = pname.replace(".", "__")
    shape = next(s for n, k, s in cm.names if n == pname)
    mu = cm.mu[key].detach().cpu().numpy()
    idx = getattr(cm, f"idx_{key}").cpu().numpy()
    sign = getattr(cm, f"sign_{key}").cpu().numpy()
    signed = ((idx + 1) * np.where(sign >= 0, 1, -1)).astype(np.int8)
    return mu, transform(signed.reshape(shape))


def export_mlp(model, path, fmt="float32", eps=None, name="model"):
    """Plain tanh MLP (the dense baselines)."""
    layers = [(m.weight.detach().cpu().numpy(),
               m.bias.detach().cpu().numpy() if m.bias is not None else None)
              for m in model.modules() if isinstance(m, torch.nn.Linear)]
    dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    L, maxw = len(layers), max(dims)

    em = _Emitter(fmt, eps)
    W = [em.add(f"{name}_w{i}", w.T.ravel()) for i, (w, _) in enumerate(layers)]
    B = [em.add_bias(f"{name}_b{i}", b) for i, (_, b) in enumerate(layers)]

    body = []
    for l, (w, _) in enumerate(layers):
        ni, no = dims[l], dims[l + 1]
        act = "acc" if l == L - 1 else "tanhf(acc)"
        body.append(f"    for (j = 0; j < {no}; j++) {{")
        body.append(f"        float acc = {B[l]('j')};")
        body.append(f"        for (i = 0; i < {ni}; i++) acc += a[i] * {W[l](f'i * {no} + j')};")
        body.append(f"        z[j] = {act};")
        body.append("    }")
        body.append(f"    for (j = 0; j < {no}; j++) a[j] = z[j];")

    src = ["/* generated by export_c.py -- do not edit */", "#include <math.h>",
           "#include <stdint.h>", ""] + em.decls
    src.append(f"/* {name}: {'x'.join(map(str, dims))} tanh MLP, {fmt} weights */")
    src.append(f"void {name}_forward(const float *in, float *out) {{")
    src.append(f"    static float a[{maxw}], z[{maxw}];")
    src.append("    int i, j;")
    src.append(f"    for (i = 0; i < {dims[0]}; i++) a[i] = in[i];")
    src.extend(body)
    src.append(f"    for (i = 0; i < {dims[-1]}; i++) out[i] = a[i];")
    src.append("}")
    open(path, "w").write("\n".join(src))

    return {"format": fmt, "dims": dims, "weight_bytes": em.weight_bytes,
            "clusters": em.n_clusters or None,
            "params": int(sum(w.size + (b.size if b is not None else 0) for w, b in layers)),
            "activation_bytes": maxw * 2 * 4}


def export_psinn_burgers(model, path, fmt="float32", eps=None, name="psinn",
                         clustered_model=None):
    """Structured Psi-NN (Burgers). Weight sharing across branches is written out
    explicitly -- that tying is the architecture, and it is what preserves the
    odd-in-x symmetry on the device exactly as it does in PyTorch.

    clustered_model: a distilled, centroid-retrained ClusteredModel (exp4). When given
    with fmt="clustered", its centroids are shipped verbatim, so the device runs the
    model whose accuracy exp4 actually validated -- not a fresh post-hoc clustering."""
    src_model = clustered_model.base if clustered_model is not None else model
    if clustered_model is not None:
        recon = {k: v.detach().cpu().numpy()
                 for k, v in clustered_model.reconstructed().items()}
        sd = recon
    else:
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    n = src_model.fc1_1.out_features

    em = _Emitter(fmt, eps)
    flat = lambda a: a.ravel()
    trans = lambda a: a.T.ravel()

    def W(pname, arrname, transform):
        if fmt == "clustered" and clustered_model is not None:
            mu, idx = cluster_tensor_from(clustered_model, pname, transform)
            return em.add_precomputed_cluster(arrname, mu, idx)
        return em.add(arrname, transform(sd[pname]))

    w11 = W("fc1_1.weight", f"{name}_fc1_1_w", flat)
    b11 = em.add_bias(f"{name}_fc1_1_b", sd["fc1_1.bias"])
    w13 = W("fc1_3.weight", f"{name}_fc1_3_w", flat)
    w21 = W("fc2_1.weight", f"{name}_fc2_1_w", trans)
    b21 = em.add_bias(f"{name}_fc2_1_b", sd["fc2_1.bias"])
    w22 = W("fc2_2.weight", f"{name}_fc2_2_w", trans)
    w23 = W("fc2_3.weight", f"{name}_fc2_3_w", trans)
    b23 = em.add_bias(f"{name}_fc2_3_b", sd["fc2_3.bias"])
    w31 = W("fc3_1.weight", f"{name}_fc3_1_w", trans)
    w32 = W("fc3_2.weight", f"{name}_fc3_2_w", trans)
    w41 = W("fc4_1.weight", f"{name}_fc4_1_w", flat)
    b41 = em.add_bias(f"{name}_fc4_1_b", sd["fc4_1.bias"])

    src = ["/* generated by export_c.py -- structured Psi-NN (Burgers) */",
           "#include <math.h>", "#include <stdint.h>", "",
           f"#define PSINN_N {n}", ""] + em.decls
    src.append(f"""
/* input = (t, x). The tied weights make u(t,-x) = -u(t,x) hold exactly, in fp32,
   on the device -- the symmetry is structural, not learned. ({fmt} weights) */
void {name}_forward(const float *in, float *out) {{
    const float t = in[0], x = in[1];
    /* static, like the dense kernel's buffers: activation memory must land in .bss so
       that arm-none-eabi-size counts it. Stack-allocated locals are NOT counted by
       `size`, and comparing a stack-allocated kernel against a statically allocated one
       would understate this model's RAM. */
    static float u1_1[PSINN_N], u1_2[PSINN_N];
    static float u2_1[PSINN_N], u2_2[PSINN_N], u2_3[PSINN_N];
    static float u3_1[2 * PSINN_N];
    int i, j;

    for (i = 0; i < PSINN_N; i++) {{
        const float a = {w11('i')} * t + {b11('i')};
        const float b = {w13('i')} * x;
        u1_1[i] = tanhf(a + b);
        u1_2[i] = tanhf(a - b);
    }}
    for (j = 0; j < PSINN_N; j++) {{
        float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (i = 0; i < PSINN_N; i++) {{
            const float a1 = {w21('i * PSINN_N + j')};
            const float a2 = {w22('i * PSINN_N + j')};
            const float a3 = {w23('i * PSINN_N + j')};
            s1 += u1_1[i] * a1 + u1_2[i] * a3;
            s2 += u1_1[i] * a3 + u1_2[i] * a1;
            s3 += (u1_1[i] - u1_2[i]) * a2;
        }}
        const float bb = {b21('j')} + {b23('j')};
        u2_1[j] = tanhf(s1 + bb);
        u2_2[j] = tanhf(s2 + bb);
        u2_3[j] = tanhf(s3);
    }}
    for (j = 0; j < 2 * PSINN_N; j++) {{
        float s = 0.0f;
        for (i = 0; i < PSINN_N; i++) {{
            s += (u2_1[i] - u2_2[i]) * {w31('i * 2 * PSINN_N + j')};
            s += u2_3[i] * {w32('i * 2 * PSINN_N + j')};
        }}
        u3_1[j] = tanhf(s);
    }}
    float o = {b41('0')};
    for (i = 0; i < 2 * PSINN_N; i++) o += u3_1[i] * {w41('i')};
    out[0] = o;
}}
""")
    open(path, "w").write("\n".join(src))
    params = int(sum(v.size for v in sd.values()))
    return {"format": fmt, "weight_bytes": em.weight_bytes,
            "clusters": em.n_clusters or None, "params": params,
            "activation_bytes": (5 * n + 2 * n) * 4}
