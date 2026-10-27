"""Linear-work recurrent scan for J-BRAIN v2 (replaces the O(L^2) matrix form of the original architecture_v2.py).

Recurrence (exactly the one in architecture_v2.JBRHolographicRecurrentCore; per batch b and head h, the D channels of a head share
one transition coefficient):

    h_t = a_t * h_{t-1} + v_t,    a_t = gamma_t * exp(i * delta_phi_t)   (complex scalar per head and step)
                                  v_t = val_r[t] + i * val_i[t]          (complex vector, D channels),        h_0 = initial state

Shapes, as produced by JBRHolographicRecurrentCore.forward: gamma, delta_phi [B, H, L]; val_r, val_i [B, H, L, D]; h0 [B, H, D].
Inputs may be non-contiguous views (the forward passes transposed views of [B, L, H, *] tensors).

Implementations (environment variable JBRAIN_SCAN, or use_impl() for a scoped override):

  reference  The original O(L^2) code of architecture_v2.py, verbatim (reference_scan). Kept for comparisons. Known defects, kept as is:
             a single pass longer than ~843 steps overflows to NaN when gamma is near its 0.9 floor (exp(C_t - C_s) is formed for
             s > t before the causal mask is applied, and inf * 0 = NaN), and under bf16 autocast its four matmuls run in bf16.
  fallback   Pure PyTorch, chunked: T x T decay-rotation matrices inside chunks of T = 32 steps (masked before exp, so nothing
             overflows) and the chunk-to-chunk recurrence solved recursively by the same routine. Work O(B*H*L*T*D), memory
             O(B*H*L*T); no tensor grows quadratically in L. Values in float32 (float64 if any input is float64), per-head
             decay/phase bookkeeping in float64 (float32 on MPS, which lacks float64), autocast disabled. Any device. Slower than
             the reference for short sequences (launch-bound), much faster for long ones.
  triton     Tiled associative scan, forward and backward kernels. Each program owns one (batch, head, 16-channel block) and walks
             the sequence in tiles of 16 steps: a tree scan inside the tile (tl.associative_scan over affine pairs), then the state
             is carried sequentially to the next tile. Work O(B*H*L*D). Parallel depth is O((L/16) * log2 16) per program, i.e.
             still linear in L; this implementation does NOT achieve O(log L) depth. float32 accumulation for float32, bfloat16
             and float16 inputs. CUDA only.
  auto       (default) triton when it is usable (CUDA tensors, Triton importable, supported dtype and shape, and a one-time
             numerical self-test passed on the device); otherwise the reference for short sequences (L <= 256 with float32/float64
             inputs, where it is faster and overflow-free), else the fallback.

Outputs are float32 for float32/bfloat16/float16 inputs (float64 for float64 inputs) on the triton and fallback paths. The reference
keeps its original dtype behaviour (bfloat16 matmul outputs under bf16 autocast).

What this module does not do: it does not parallelise autoregressive generation. During decoding the next token, and therefore every
coefficient at the next position, depends on the current state; architecture_v2 keeps its single-token update (the
`L == 1 and past_state is not None` branch of JBRHolographicRecurrentCore.forward), which never calls this module.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import warnings

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except Exception as _e:  # not installed, or a broken installation
    triton = tl = None
    _TRITON_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

log = logging.getLogger("jbrain_scan")

IMPLS = ("auto", "triton", "fallback", "reference")
ENV_VAR = "JBRAIN_SCAN"
TILE_T = 16                   # time steps per Triton tile 
BLOCK_D = 16                  # channels per Triton program
NUM_WARPS = 1
FALLBACK_CHUNK = 32           # chunk length of the PyTorch fallback
AUTO_REFERENCE_MAX_LEN = 256  # auto without Triton: up to here the old code is faster than the fallback on CPU (measured; 256 is also
                              # the edge backend's prefill chunk) and cannot overflow (that needs ~843 steps)
_INT32_MAX = 2**31 - 1        # the kernels index with int32 offsets

CALL_COUNTS = {"reference": 0, "fallback": 0, "triton": 0}   # incremented on every call; used by the decode-path regression test
# Process-wide, deliberately not thread-local: with gradient checkpointing the autograd engine re-runs the layer forward (and so
# this module) during backward, on its own worker threads, and that recomputation must select the same implementation.
_override = {"name": None}
_selftest_cache: dict = {}
_logged: set = set()


# selection
def requested_impl() -> str:
    """The implementation requested by use_impl() (innermost) or by JBRAIN_SCAN (default "auto")."""
    name = _override["name"] or os.environ.get(ENV_VAR, "auto")
    name = name.strip().lower()
    if name not in IMPLS:
        raise ValueError(f"{ENV_VAR}={name!r} is not one of {IMPLS}")
    return name


@contextlib.contextmanager
def use_impl(name: str):
    """Process-wide temporary override of JBRAIN_SCAN, e.g. `with jbrain_scan.use_impl("reference"): ...` (for tests and benchmarks).
    With gradient checkpointing, keep the backward pass inside the same block: the checkpointed layers are recomputed during
    backward and must use the same implementation (otherwise torch.utils.checkpoint raises CheckpointError)."""
    if name not in IMPLS:
        raise ValueError(f"implementation {name!r} is not one of {IMPLS}")
    previous = _override["name"]
    _override["name"] = name
    try:
        yield
    finally:
        _override["name"] = previous


def reset_call_counts() -> None:
    for k in CALL_COUNTS:
        CALL_COUNTS[k] = 0


def _max_offset(t: torch.Tensor) -> int:
    return sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride()) if s > 0)


def triton_usable(gamma, delta_phi, val_r, val_i, h0_r=None, h0_i=None) -> tuple[bool, str]:
    """Whether the Triton path can run these inputs, and why not if it cannot. Runs the one-time self-test on first use."""
    if triton is None:
        return False, f"Triton is not importable ({_TRITON_IMPORT_ERROR})"
    if not val_r.is_cuda:
        return False, f"tensors are on {val_r.device.type}, not CUDA"
    tensors = [gamma, delta_phi, val_r, val_i] + ([h0_r, h0_i] if h0_r is not None else [])
    if any(t.device != val_r.device for t in tensors):
        return False, "inputs are on different devices"
    if any(t.dtype not in (torch.float32, torch.bfloat16, torch.float16) for t in tensors):
        return False, "dtype is not float32/bfloat16/float16"
    D = val_r.shape[-1]
    if D % BLOCK_D != 0:
        return False, f"head dimension {D} is not a multiple of {BLOCK_D}"
    if val_r.numel() > _INT32_MAX or any(_max_offset(t) > _INT32_MAX for t in tensors):
        return False, "tensor too large for int32 indexing"
    ok, reason = _triton_selftest(val_r.device)
    return (True, "ok") if ok else (False, reason)


def select_impl(gamma, delta_phi, val_r, val_i, h0_r=None, h0_i=None, impl: str | None = None) -> tuple[str, str]:
    """(implementation, reason) for these inputs under the requested mode."""
    name = impl if impl is not None else requested_impl()
    if name not in IMPLS:
        raise ValueError(f"implementation {name!r} is not one of {IMPLS}")
    if name in ("reference", "fallback"):
        return name, "requested"
    ok, why = triton_usable(gamma, delta_phi, val_r, val_i, h0_r, h0_i)
    if name == "triton":
        if not ok:
            raise RuntimeError(f"{ENV_VAR}=triton was requested but the Triton scan cannot run here: {why}")
        return "triton", "requested"
    if ok:
        return "triton", "auto: Triton usable"
    L = val_r.shape[-2]
    exact = all(t.dtype in (torch.float32, torch.float64) for t in (gamma, delta_phi, val_r, val_i))
    if L <= AUTO_REFERENCE_MAX_LEN and exact:
        return "reference", f"auto: {why}; short sequence"
    return "fallback", f"auto: {why}"


def _note(choice: str, reason: str, device) -> None:
    key = (choice, reason.split(";")[0], str(device))
    if key not in _logged:
        _logged.add(key)
        log.info("jbrain_scan: %s on %s (%s)", choice, device, reason)


def _check(gamma, delta_phi, val_r, val_i, h0_r, h0_i) -> None:
    if val_r.dim() != 4 or val_i.shape != val_r.shape:
        raise ValueError(f"val_r/val_i must both be [B, H, L, D]; got {tuple(val_r.shape)} and {tuple(val_i.shape)}")
    if gamma.shape != val_r.shape[:3] or delta_phi.shape != val_r.shape[:3]:
        raise ValueError(f"gamma/delta_phi must be [B, H, L] = {tuple(val_r.shape[:3])}; got {tuple(gamma.shape)}, {tuple(delta_phi.shape)}")
    if (h0_r is None) != (h0_i is None):
        raise ValueError("pass both h0_r and h0_i, or neither")
    if h0_r is not None and (h0_r.shape != val_r.shape[:2] + val_r.shape[3:] or h0_i.shape != h0_r.shape):
        raise ValueError(f"h0_r/h0_i must be [B, H, D] = {tuple(val_r.shape[:2] + val_r.shape[3:])}")


#public API
def scan(gamma, delta_phi, val_r, val_i, h0_r=None, h0_i=None, impl: str | None = None):
    """All states h_1..h_L of h_t = gamma_t e^{i delta_phi_t} h_{t-1} + (val_r + i val_i)_t, starting from h_0 = h0_r + i h0_i
    (zero when omitted). Returns (h_r, h_i), each [B, H, L, D]. Differentiable with respect to every input."""
    _check(gamma, delta_phi, val_r, val_i, h0_r, h0_i)
    choice, reason = select_impl(gamma, delta_phi, val_r, val_i, h0_r, h0_i, impl)
    CALL_COUNTS[choice] += 1
    _note(choice, reason, val_r.device)
    if choice == "reference":
        return _reference_with_h0(gamma, delta_phi, val_r, val_i, h0_r, h0_i)
    if choice == "fallback":
        return fallback_scan(gamma, delta_phi, val_r, val_i, h0_r, h0_i)
    return _triton_scan(gamma, delta_phi, val_r, val_i, h0_r, h0_i)


def complex_phase_parallel_scan(gamma_t, delta_phi_t, val_r, val_i, impl: str | None = None):
    """Drop-in for the original JBRHolographicRecurrentCore.complex_phase_parallel_scan: zero initial state; returns
    (B_r, B_i, C, Phi) with B_r, B_i [B, H, L, D] the states and C = cumsum(log(gamma + 1e-8)), Phi = cumsum(delta_phi) [B, H, L],
    computed exactly as before (the caller uses C and Phi to add the carried state of a previous chunk)."""
    _check(gamma_t, delta_phi_t, val_r, val_i, None, None)
    choice, reason = select_impl(gamma_t, delta_phi_t, val_r, val_i, impl=impl)
    CALL_COUNTS[choice] += 1
    _note(choice, reason, val_r.device)
    if choice == "reference":
        return reference_scan(gamma_t, delta_phi_t, val_r, val_i)
    if choice == "fallback":
        B_r, B_i = fallback_scan(gamma_t, delta_phi_t, val_r, val_i)
    else:
        B_r, B_i = _triton_scan(gamma_t, delta_phi_t, val_r, val_i)
    log_gamma = torch.log(gamma_t + 1e-8)
    C = torch.cumsum(log_gamma, dim=-1)
    Phi = torch.cumsum(delta_phi_t, dim=-1)
    return B_r, B_i, C, Phi


#reference
def reference_scan(gamma_t, delta_phi_t, val_r, val_i):
    """The original body of architecture_v2.JBRHolographicRecurrentCore.complex_phase_parallel_scan, verbatim (O(L^2))."""
    B, H, L, D = val_r.shape

    log_gamma = torch.log(gamma_t + 1e-8)
    C = torch.cumsum(log_gamma, dim=-1) # [B, H, L]

    Phi = torch.cumsum(delta_phi_t, dim=-1) # [B, H, L]

    C_diff = C.unsqueeze(-1) - C.unsqueeze(-2)       # C_t - C_s
    Phi_diff = Phi.unsqueeze(-1) - Phi.unsqueeze(-2) # Phi_t - Phi_s

    decay_matrix = torch.exp(C_diff)

    # construct the cosine and sine matrices of the unitary operator
    M_cos = decay_matrix * torch.cos(Phi_diff)
    M_sin = decay_matrix * torch.sin(Phi_diff)

    causal_mask = torch.tril(torch.ones(L, L, device=val_r.device)).view(1, 1, L, L)
    M_cos = M_cos * causal_mask
    M_sin = M_sin * causal_mask

    B_r = torch.matmul(M_cos, val_r) - torch.matmul(M_sin, val_i)
    B_i = torch.matmul(M_sin, val_r) + torch.matmul(M_cos, val_i)

    return B_r, B_i, C, Phi


def _reference_with_h0(gamma, delta_phi, val_r, val_i, h0_r, h0_i):
    """reference_scan plus the carried-state term of architecture_v2.JBRHolographicRecurrentCore.forward (same operations)."""
    B_r, B_i, C, Phi = reference_scan(gamma, delta_phi, val_r, val_i)
    if h0_r is None:
        return B_r, B_i
    decay_past = torch.exp(C).unsqueeze(-1)
    cos_past = torch.cos(Phi).unsqueeze(-1)
    sin_past = torch.sin(Phi).unsqueeze(-1)
    h_r_p, h_i_p = h0_r.unsqueeze(2), h0_i.unsqueeze(2)
    return (B_r + decay_past * (cos_past * h_r_p - sin_past * h_i_p),
            B_i + decay_past * (sin_past * h_r_p + cos_past * h_i_p))


# fallback
def _no_autocast(device_type: str):
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:  # a device type without autocast support
        return contextlib.nullcontext()


def _wrap_angle(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def _block_states(c, p, vr, vi):
    """States of one block from a zero state. c, p: [..., n] inclusive cumulative log-decay and phase; vr, vi: [..., n, D].
    Uses W[t, s] = exp(c_t - c_s) e^{i (p_t - p_s)} for s <= t; the s > t entries are set to -inf BEFORE exp, so they are exactly 0.
    c and p may be in a wider dtype than the values (float64 bookkeeping): the weights are cast to the values' dtype for the matmul."""
    n = c.shape[-1]
    tri = torch.ones(n, n, dtype=torch.bool, device=c.device).tril()
    E = torch.exp((c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~tri, float("-inf")))
    pd = p.unsqueeze(-1) - p.unsqueeze(-2)
    Wr, Wi = (E * torch.cos(pd)).to(vr.dtype), (E * torch.sin(pd)).to(vr.dtype)
    return Wr @ vr - Wi @ vi, Wi @ vr + Wr @ vi


def _apply_carry(c, p, hr, hi, car_r, car_i):
    """h += (prod of a up to t, i.e. exp(c_t) e^{i p_t}) * carried state."""
    e = torch.exp(c)
    qr, qi = (e * torch.cos(p)).unsqueeze(-1).to(hr.dtype), (e * torch.sin(p)).unsqueeze(-1).to(hr.dtype)
    return hr + qr * car_r - qi * car_i, hi + qi * car_r + qr * car_i


def _chunked(log_g, phi, vr, vi, h0r, h0i, T):
    L = log_g.shape[-1]
    if L <= T:
        c, p = log_g.cumsum(-1), phi.cumsum(-1)
        hr, hi = _block_states(c, p, vr, vi)
        if h0r is not None:
            hr, hi = _apply_carry(c, p, hr, hi, h0r.unsqueeze(-2), h0i.unsqueeze(-2))
        return hr, hi
    B, H, _, D = vr.shape
    pad = (-L) % T
    if pad:   # identity steps (log gamma = 0, phase 0, value 0) at the end; they are sliced off below
        log_g, phi = F.pad(log_g, (0, pad)), F.pad(phi, (0, pad))
        vr, vi = F.pad(vr, (0, 0, 0, pad)), F.pad(vi, (0, 0, 0, pad))
    N = (L + pad) // T
    c = log_g.reshape(B, H, N, T).cumsum(-1)                  # local cumulative log-decay (<= 0) and phase (|p| <= T * pi)
    p = phi.reshape(B, H, N, T).cumsum(-1)
    hr, hi = _block_states(c, p, vr.reshape(B, H, N, T, D), vi.reshape(B, H, N, T, D))
    # chunk-level recurrence S_k = A_k S_{k-1} + s_k (A_k = whole-chunk transition, s_k = chunk-final local state), S_0 = h0:
    # the same problem with N steps, solved recursively (log-decays stay finite; phases are wrapped to (-pi, pi]).
    S_r, S_i = _chunked(c[..., -1], _wrap_angle(p[..., -1]), hr[..., -1, :], hi[..., -1, :], h0r, h0i, T)
    first_r = hr.new_zeros(B, H, 1, D) if h0r is None else h0r.unsqueeze(2)
    first_i = hr.new_zeros(B, H, 1, D) if h0i is None else h0i.unsqueeze(2)
    car_r = torch.cat([first_r, S_r[:, :, :-1]], 2).unsqueeze(3)     # state entering chunk k
    car_i = torch.cat([first_i, S_i[:, :, :-1]], 2).unsqueeze(3)
    hr, hi = _apply_carry(c, p, hr, hi, car_r, car_i)
    return hr.reshape(B, H, N * T, D)[:, :, :L], hi.reshape(B, H, N * T, D)[:, :, :L]


def fallback_scan(gamma, delta_phi, val_r, val_i, h0_r=None, h0_i=None, chunk: int = FALLBACK_CHUNK):
    """Pure PyTorch chunked scan (any device). Requires gamma > 0 (log gamma is used).
    Values and states are computed in float32 (float64 if any input is float64). The per-head bookkeeping (cumulative log-decay
    and phase, their pairwise differences, exp/cos/sin) is done in float64, except on MPS, which has no float64: forming
    differences of float32 cumulative phases (up to chunk * pi) costs up to ~1e-5 rad and made the fallback up to ~10x less
    accurate than the float32 step-by-step recurrence with fast decay and large rotations; in float64 it is more accurate than
    that baseline. These tensors have no channel dimension, so the float64 part is cheap."""
    tensors = [gamma, delta_phi, val_r, val_i] + ([h0_r, h0_i] if h0_r is not None else [])
    cdt = torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32
    sdt = torch.float32 if val_r.device.type == "mps" else torch.float64
    with _no_autocast(val_r.device.type):
        return _chunked(torch.log(gamma.to(sdt)), delta_phi.to(sdt), val_r.to(cdt), val_i.to(cdt),
                        None if h0_r is None else h0_r.to(cdt), None if h0_i is None else h0_i.to(cdt), chunk)


# Triton
if triton is not None:
    @triton.jit
    def _combine(ar1, ai1, br1, bi1, ar2, ai2, br2, bi2):
        # affine pairs, earlier (a1, b1) then later (a2, b2):  x -> a2 (a1 x + b1) + b2 = (a2 a1) x + (a2 b1 + b2)
        return (ar2 * ar1 - ai2 * ai1,
                ar2 * ai1 + ai2 * ar1,
                ar2 * br1 - ai2 * bi1 + br2,
                ar2 * bi1 + ai2 * br1 + bi2)

    @triton.jit
    def _scan_fwd_kernel(G, P, VR, VI, H0R, H0I, HR, HI, L, NH,
                         sg_b, sg_h, sg_l, sv_b, sv_h, sv_l, so_b, so_h, so_l,
                         HAS_H0: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, T: tl.constexpr):
        pid = tl.program_id(0)                       # b * NH + h
        pd = tl.program_id(1)                        # channel block
        b = pid // NH
        h = pid % NH
        offs_d = pd * BLOCK + tl.arange(0, BLOCK)
        rows = tl.arange(0, T)
        g_base = G + b * sg_b + h * sg_h
        p_base = P + b * sg_b + h * sg_h
        v_off = b * sv_b + h * sv_h
        o_off = b * so_b + h * so_h
        if HAS_H0:
            cr = tl.load(H0R + pid * D + offs_d).to(tl.float32)
            ci = tl.load(H0I + pid * D + offs_d).to(tl.float32)
        else:
            cr = tl.zeros([BLOCK], dtype=tl.float32)
            ci = tl.zeros([BLOCK], dtype=tl.float32)
        for t0 in range(0, L, T):
            t = t0 + rows
            mt = t < L
            g = tl.load(g_base + t * sg_l, mask=mt, other=1.0).to(tl.float32)      # padding rows: identity (a = 1, v = 0)
            ph = tl.load(p_base + t * sg_l, mask=mt, other=0.0).to(tl.float32)
            ar = tl.broadcast_to((g * tl.cos(ph))[:, None], (T, BLOCK))
            ai = tl.broadcast_to((g * tl.sin(ph))[:, None], (T, BLOCK))
            vptr = v_off + t[:, None] * sv_l + offs_d[None, :]
            vr = tl.load(VR + vptr, mask=mt[:, None], other=0.0).to(tl.float32)
            vi = tl.load(VI + vptr, mask=mt[:, None], other=0.0).to(tl.float32)
            Ar, Ai, Br, Bi = tl.associative_scan((ar, ai, vr, vi), 0, _combine)    # inclusive prefix within the tile
            hr = Ar * cr[None, :] - Ai * ci[None, :] + Br                           # h_t = A_t * carry + B_t
            hi = Ai * cr[None, :] + Ar * ci[None, :] + Bi
            optr = o_off + t[:, None] * so_l + offs_d[None, :]
            tl.store(HR + optr, hr, mask=mt[:, None])
            tl.store(HI + optr, hi, mask=mt[:, None])
            last = (rows == T - 1)[:, None]                                        # carry = last row of the tile
            cr = tl.sum(tl.where(last, hr, 0.0), axis=0)
            ci = tl.sum(tl.where(last, hi, 0.0), axis=0)

    @triton.jit
    def _scan_bwd_kernel(G, P, HR, HI, H0R, H0I, DHR, DHI, DVR, DVI, DAR, DAI, L, NH, NDB,
                         sg_b, sg_h, sg_l, so_b, so_h, so_l, sd_b, sd_h, sd_l,
                         HAS_H0: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, T: tl.constexpr):
        # adjoint: g_t = dh_t + conj(a_{t+1}) g_{t+1} (the same scan in reverse time); dv_t = g_t; da_t = sum_d g_t conj(h_{t-1})
        pid = tl.program_id(0)
        pd = tl.program_id(1)
        b = pid // NH
        h = pid % NH
        offs_d = pd * BLOCK + tl.arange(0, BLOCK)
        rows = tl.arange(0, T)
        g_base = G + b * sg_b + h * sg_h
        p_base = P + b * sg_b + h * sg_h
        o_off = b * so_b + h * so_h
        d_off = b * sd_b + h * sd_h
        if HAS_H0:
            h0r = tl.load(H0R + pid * D + offs_d).to(tl.float32)
            h0i = tl.load(H0I + pid * D + offs_d).to(tl.float32)
        cr = tl.zeros([BLOCK], dtype=tl.float32)
        ci = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, L, T):
            t = L - 1 - k - rows                     # tile rows in descending time order
            mt = t >= 0
            mn = mt & (t + 1 < L)
            g = tl.load(g_base + (t + 1) * sg_l, mask=mn, other=0.0).to(tl.float32)
            ph = tl.load(p_base + (t + 1) * sg_l, mask=mn, other=0.0).to(tl.float32)
            alr = tl.broadcast_to((g * tl.cos(ph))[:, None], (T, BLOCK))            # conj(a_{t+1}); 0 beyond the end
            ali = tl.broadcast_to((-g * tl.sin(ph))[:, None], (T, BLOCK))
            dptr = d_off + t[:, None] * sd_l + offs_d[None, :]
            dhr = tl.load(DHR + dptr, mask=mt[:, None], other=0.0).to(tl.float32)
            dhi = tl.load(DHI + dptr, mask=mt[:, None], other=0.0).to(tl.float32)
            Ar, Ai, Br, Bi = tl.associative_scan((alr, ali, dhr, dhi), 0, _combine)
            gr = Ar * cr[None, :] - Ai * ci[None, :] + Br
            gi = Ai * cr[None, :] + Ar * ci[None, :] + Bi
            optr = o_off + t[:, None] * so_l + offs_d[None, :]
            tl.store(DVR + optr, gr, mask=mt[:, None])
            tl.store(DVI + optr, gi, mask=mt[:, None])
            mp = mt & (t >= 1)
            pptr = o_off + (t - 1)[:, None] * so_l + offs_d[None, :]
            hpr = tl.load(HR + pptr, mask=mp[:, None], other=0.0)
            hpi = tl.load(HI + pptr, mask=mp[:, None], other=0.0)
            if HAS_H0:
                first = (t == 0)[:, None]
                hpr = tl.where(first, h0r[None, :], hpr)
                hpi = tl.where(first, h0i[None, :], hpi)
            dar = tl.sum(gr * hpr + gi * hpi, axis=1)
            dai = tl.sum(gi * hpr - gr * hpi, axis=1)
            aptr = (pid * NDB + pd) * L + t
            tl.store(DAR + aptr, dar, mask=mt)
            tl.store(DAI + aptr, dai, mask=mt)
            last = (rows == T - 1)[:, None]
            cr = tl.sum(tl.where(last, gr, 0.0), axis=0)
            ci = tl.sum(tl.where(last, gi, 0.0), axis=0)


def _alloc_like_states(B, L, H, D, device):
    # [B, L, H, D] memory presented as [B, H, L, D]: the layout JBRHolographicRecurrentCore.forward transposes back to
    return torch.empty(B, L, H, D, device=device, dtype=torch.float32).transpose(1, 2)


def _triton_forward(gamma, delta_phi, val_r, val_i, h0_r, h0_i):
    B, H, L, D = val_r.shape
    hr, hi = _alloc_like_states(B, L, H, D, val_r.device), _alloc_like_states(B, L, H, D, val_r.device)
    if L == 0 or B * H == 0:
        return hr, hi
    has_h0 = h0_r is not None
    sg, sv, so = gamma.stride(), val_r.stride(), hr.stride()
    with torch.cuda.device(val_r.device):
        _scan_fwd_kernel[(B * H, D // BLOCK_D)](
            gamma, delta_phi, val_r, val_i, h0_r if has_h0 else hr, h0_i if has_h0 else hr, hr, hi, L, H,
            sg[0], sg[1], sg[2], sv[0], sv[1], sv[2], so[0], so[1], so[2],
            HAS_H0=has_h0, D=D, BLOCK=BLOCK_D, T=TILE_T, num_warps=NUM_WARPS)
    return hr, hi


def _triton_backward(gamma, delta_phi, hr, hi, h0_r, h0_i, dhr, dhi):
    """float32 gradients (d_gamma, d_delta_phi, d_val_r, d_val_i, d_h0_r, d_h0_i) for the saved forward states hr, hi."""
    B, H, L, D = hr.shape
    if dhr.stride(-1) != 1 or dhi.stride() != dhr.stride():
        dhr, dhi = dhr.contiguous(), dhi.contiguous()
    dvr, dvi = _alloc_like_states(B, L, H, D, hr.device), _alloc_like_states(B, L, H, D, hr.device)
    if hr.stride() != dvr.stride():              # the kernel reads h and writes dv with the same strides
        hr, hi = hr.transpose(1, 2).contiguous().transpose(1, 2), hi.transpose(1, 2).contiguous().transpose(1, 2)
    NDB = D // BLOCK_D
    dar = torch.zeros(B * H, NDB, L, device=hr.device, dtype=torch.float32)
    dai = torch.zeros_like(dar)
    has_h0 = h0_r is not None
    if L > 0 and B * H > 0:
        sg, so, sd = gamma.stride(), hr.stride(), dhr.stride()
        with torch.cuda.device(hr.device):
            _scan_bwd_kernel[(B * H, NDB)](
                gamma, delta_phi, hr, hi, h0_r if has_h0 else hr, h0_i if has_h0 else hr, dhr, dhi, dvr, dvi, dar, dai, L, H, NDB,
                sg[0], sg[1], sg[2], so[0], so[1], so[2], sd[0], sd[1], sd[2],
                HAS_H0=has_h0, D=D, BLOCK=BLOCK_D, T=TILE_T, num_warps=NUM_WARPS)
    with _no_autocast("cuda"):
        da_r = dar.sum(1).view(B, H, L)
        da_i = dai.sum(1).view(B, H, L)
        g32, p32 = gamma.float(), delta_phi.float()
        cs, sn = torch.cos(p32), torch.sin(p32)
        d_gamma = da_r * cs + da_i * sn                        # a = gamma (cos phi, sin phi)
        d_phi = g32 * (da_i * cs - da_r * sn)
        d_h0r = d_h0i = None
        if has_h0 and L > 0:                                   # dh0 = conj(a_1) g_1
            ar0 = (g32[:, :, 0] * cs[:, :, 0]).unsqueeze(-1)
            ai0 = (g32[:, :, 0] * sn[:, :, 0]).unsqueeze(-1)
            g0r, g0i = dvr[:, :, 0], dvi[:, :, 0]
            d_h0r = ar0 * g0r + ai0 * g0i
            d_h0i = ar0 * g0i - ai0 * g0r
        elif has_h0:
            d_h0r, d_h0i = torch.zeros_like(h0_r), torch.zeros_like(h0_i)
    return d_gamma, d_phi, dvr, dvi, d_h0r, d_h0i


class _TritonScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma, delta_phi, val_r, val_i, h0_r, h0_i):
        hr, hi = _triton_forward(gamma, delta_phi, val_r, val_i, h0_r, h0_i)
        ctx.save_for_backward(gamma, delta_phi, hr, hi, h0_r, h0_i)
        ctx.dtypes = (gamma.dtype, delta_phi.dtype, val_r.dtype, val_i.dtype, None if h0_r is None else h0_r.dtype)
        return hr, hi

    @staticmethod
    @once_differentiable
    def backward(ctx, dhr, dhi):
        gamma, delta_phi, hr, hi, h0_r, h0_i = ctx.saved_tensors
        dhr = torch.zeros_like(hr) if dhr is None else dhr
        dhi = torch.zeros_like(hi) if dhi is None else dhi
        dg, dp, dvr, dvi, dh0r, dh0i = _triton_backward(gamma, delta_phi, hr, hi, h0_r, h0_i, dhr, dhi)
        tg, tp, tvr, tvi, th0 = ctx.dtypes
        return (dg.to(tg), dp.to(tp), dvr.to(tvr), dvi.to(tvi),
                None if dh0r is None else dh0r.to(th0), None if dh0i is None else dh0i.to(th0))


def _triton_scan(gamma, delta_phi, val_r, val_i, h0_r=None, h0_i=None):
    # layout normalisation happens outside the autograd Function, so gradients flow back through these copies
    if val_r.stride(-1) != 1 or val_i.stride(-1) != 1 or val_r.stride() != val_i.stride():
        val_r, val_i = val_r.contiguous(), val_i.contiguous()
    if gamma.stride() != delta_phi.stride():
        gamma, delta_phi = gamma.contiguous(), delta_phi.contiguous()
    if h0_r is not None:
        h0_r, h0_i = h0_r.float().contiguous(), h0_i.float().contiguous()
    return _TritonScanFunction.apply(gamma, delta_phi, val_r, val_i, h0_r, h0_i)


# self-test
def _sequential(gamma, delta_phi, val_r, val_i, h0_r, h0_i):
    """Step-by-step recurrence (used by the self-test in float64)."""
    a_r, a_i = (gamma * torch.cos(delta_phi)).unsqueeze(-1), (gamma * torch.sin(delta_phi)).unsqueeze(-1)
    hr, hi = h0_r, h0_i
    out_r, out_i = [], []
    for t in range(val_r.shape[2]):
        hr, hi = a_r[:, :, t] * hr - a_i[:, :, t] * hi + val_r[:, :, t], a_i[:, :, t] * hr + a_r[:, :, t] * hi + val_i[:, :, t]
        out_r.append(hr)
        out_i.append(hi)
    return torch.stack(out_r, 2), torch.stack(out_i, 2)


def _sequential_adjoint(gamma, delta_phi, hr, hi, h0_r, h0_i, dhr, dhi):
    """Step-by-step adjoint of _sequential, no autograd (self-test reference)."""
    L = hr.shape[2]
    a_r, a_i = gamma * torch.cos(delta_phi), gamma * torch.sin(delta_phi)
    gr, gi = torch.zeros_like(h0_r), torch.zeros_like(h0_i)
    dvr, dvi = torch.zeros_like(hr), torch.zeros_like(hi)
    da_r, da_i = torch.zeros_like(gamma), torch.zeros_like(gamma)
    for t in range(L - 1, -1, -1):
        if t + 1 < L:
            nr, ni = a_r[:, :, t + 1].unsqueeze(-1), -a_i[:, :, t + 1].unsqueeze(-1)
            gr, gi = dhr[:, :, t] + nr * gr - ni * gi, dhi[:, :, t] + ni * gr + nr * gi
        else:
            gr, gi = dhr[:, :, t].clone(), dhi[:, :, t].clone()
        dvr[:, :, t], dvi[:, :, t] = gr, gi
        pr, pi = (hr[:, :, t - 1], hi[:, :, t - 1]) if t >= 1 else (h0_r, h0_i)
        da_r[:, :, t] = (gr * pr + gi * pi).sum(-1)
        da_i[:, :, t] = (gi * pr - gr * pi).sum(-1)
    cs, sn = torch.cos(delta_phi), torch.sin(delta_phi)
    ar0, ai0 = a_r[:, :, 0].unsqueeze(-1), a_i[:, :, 0].unsqueeze(-1)
    d_h0r = ar0 * dvr[:, :, 0] + ai0 * dvi[:, :, 0]
    d_h0i = ar0 * dvi[:, :, 0] - ai0 * dvr[:, :, 0]
    return da_r * cs + da_i * sn, gamma * (da_i * cs - da_r * sn), dvr, dvi, d_h0r, d_h0i


def _rel_max(x, ref):
    return float((x.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30))


def _device_key(device) -> str:
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


def _triton_selftest(device) -> tuple[bool, str]:
    """One-time check on each device that the compiled kernels reproduce a float64 step-by-step recurrence (forward and backward),
    so that an incompatible Triton build degrades to the fallback instead of silently producing wrong numbers."""
    key = _device_key(device)
    if key in _selftest_cache:
        return _selftest_cache[key]
    try:
        with torch.inference_mode(False), torch.no_grad(), _no_autocast("cuda"):
            gen = torch.Generator().manual_seed(1234)
            B, H, L, D = 1, 2, 37, 2 * BLOCK_D                     # L not a multiple of the tile; two channel blocks
            mk = lambda *s, scale=1.0: (scale * torch.randn(*s, generator=gen)).to(device)
            gamma = (0.9 + 0.099 * torch.rand(B, H, L, generator=gen)).to(device)
            dphi = (math.pi * (2 * torch.rand(B, H, L, generator=gen) - 1)).to(device)
            vr, vi, h0r, h0i = mk(B, H, L, D, scale=0.1), mk(B, H, L, D, scale=0.1), mk(B, H, D, scale=0.5), mk(B, H, D, scale=0.5)
            dhr, dhi = mk(B, H, L, D), mk(B, H, L, D)
            hr, hi = _triton_forward(gamma, dphi, vr, vi, h0r, h0i)
            got = _triton_backward(gamma, dphi, hr, hi, h0r, h0i, dhr, dhi)
            d = lambda t: t.double()
            rr, ri = _sequential(d(gamma), d(dphi), d(vr), d(vi), d(h0r), d(h0i))
            want = _sequential_adjoint(d(gamma), d(dphi), rr, ri, d(h0r), d(h0i), d(dhr), d(dhi))
            err_f = max(_rel_max(hr, rr), _rel_max(hi, ri))
            err_b = max(_rel_max(a, b) for a, b in zip(got, want))
            finite = all(bool(torch.isfinite(t).all()) for t in (hr, hi, *got))
        ok = finite and err_f < 1e-4 and err_b < 1e-4               # a priori fp32 bound for 37 steps is ~2e-5
        reason = f"self-test forward rel. error {err_f:.1e}, backward {err_b:.1e}" + ("" if finite else ", non-finite output")
    except Exception as e:  # compilation or launch failure on this Triton/GPU combination
        ok, reason = False, f"self-test raised {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    if not ok:
        warnings.warn(f"jbrain_scan: Triton scan disabled on {key} ({reason}); falling back", RuntimeWarning, stacklevel=3)
    _selftest_cache[key] = (ok, reason)
    return ok, reason


def triton_selftest_result(device="cuda") -> tuple[bool, str]:
    """Run (or return the cached) self-test result for a device; useful for diagnostics."""
    if triton is None:
        return False, f"Triton is not importable ({_TRITON_IMPORT_ERROR})"
    return _triton_selftest(device)
