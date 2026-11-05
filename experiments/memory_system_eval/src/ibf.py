"""Interference-Bound Field (IBF): a role-bound phasor superposition index with exact erasure and an analytic coherence test.

Codes: unit phasors in C^D derived from a hash of a string (entity id, attribute id, pointer id).
Key   kappa(e, a) = u_E(e) * Pi(u_A(a))       (elementwise product; Pi = fixed permutation: role binding)
Field F = sum_j p_j * conj(kappa_j)            (p_j = pointer code of record j)
Read  y = F * kappa_q;  s_j = <y, p_j> / D     (catalogue resonance over stored pointers)
Under H0 (no stored key coherent with the query) s_j is a sum of M*D random phasors: E|s_j|^2 = M * mean|kappa_q|^2 / D, so
r_j = |s_j| / sigma has P(r_j >= r) = exp(-r^2) (Rayleigh) and a Bonferroni bound over M pointers gives r* = sqrt(ln(M / alpha)).
An exact match contributes s = 1 (for a unit-modulus query). Updates leave F unchanged (same identity, same pointer; the payload
lives in the ledger); deletion subtracts the term exactly. Variant 'bundle' replaces binding by superposition
(kappa = (u_E + Pi u_A) / sqrt 2) for the binding test."""
from __future__ import annotations

import hashlib
import math

import numpy as np


class Codes:
    def __init__(self, D: int = 1024, seed: int = 0, max_cache: int | None = None):
        self.D = D
        self.perm = np.random.default_rng(seed).permutation(D)
        self.cache: dict[str, np.ndarray] = {}
        self.max_cache = max_cache

    def code(self, key: str, cache: bool = True) -> np.ndarray:
        c = self.cache.get(key)
        if c is None:
            h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
            c = np.exp(1j * np.random.default_rng(h).uniform(0, 2 * np.pi, self.D)).astype(np.complex64)
            if cache:
                if self.max_cache is not None and len(self.cache) >= self.max_cache:
                    self.cache.clear()                      # bounded memory (codes are recomputable from their key)
                self.cache[key] = c
        return c

    def key(self, eid: str, attr: str, mode: str = "bind") -> np.ndarray:
        e, a = self.code("E:" + eid), self.code("A:" + attr)[self.perm]
        if mode == "bind":
            return e * a
        return ((e + a) / math.sqrt(2)).astype(np.complex64)


class Field:
    def __init__(self, codes: Codes, mode: str = "bind"):
        self.c, self.mode = codes, mode
        self.F = np.zeros(codes.D, dtype=np.complex64)
        self.ids: list[str] = []
        self.plist: list[np.ndarray] = []
        self._P = None

    @property
    def P(self) -> np.ndarray:
        """Pointer catalogue [M, D], stacked lazily (writes are O(D), not O(M D))."""
        if self._P is None:
            self._P = np.stack(self.plist) if self.plist else np.zeros((0, self.c.D), dtype=np.complex64)
        return self._P

    @staticmethod
    def rid(eid, attr):
        return f"{eid}|{attr}"

    def write(self, eid: str, attr: str):
        r = self.rid(eid, attr)
        if r in self.ids:
            return False                                     # update: same identity, field unchanged
        p = self.c.code("P:" + r)
        self.F += p * np.conj(self.c.key(eid, attr, self.mode))
        self.ids.append(r)
        self.plist.append(p)
        self._P = None
        return True

    def erase(self, eid: str, attr: str):
        r = self.rid(eid, attr)
        if r not in self.ids:
            return False
        i = self.ids.index(r)
        self.F -= self.plist[i] * np.conj(self.c.key(eid, attr, self.mode))
        self.ids.pop(i)
        self.plist.pop(i)
        self._P = None
        return True

    def query_code(self, ent_w: dict, attr_w: dict) -> np.ndarray:
        q = np.zeros(self.c.D, dtype=np.complex64)
        for e, we in ent_w.items():
            for a, wa in attr_w.items():
                if we * wa > 0:
                    q += np.complex64(we * wa) * self.c.key(e, a, self.mode)
        return q

    def read(self, kq: np.ndarray):
        """Returns (record ids sorted by resonance, normalised resonances r_j, analytic sigma)."""
        M = len(self.ids)
        if M == 0 or not np.any(kq):
            return [], np.zeros(0), 0.0
        y = self.F * kq
        s = np.abs(self.P.conj() @ y) / self.c.D
        sigma = math.sqrt(M * float(np.mean(np.abs(kq) ** 2)) / self.c.D)
        r = s / sigma
        order = np.argsort(-r)
        return [self.ids[i] for i in order], r[order], sigma

    @staticmethod
    def threshold(M: int, alpha: float = 0.01) -> float:
        return math.sqrt(math.log(max(M, 1) / alpha))
