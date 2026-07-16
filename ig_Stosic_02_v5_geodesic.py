from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
geodezija — diskretni korak duž geodezije na p_t (integracija sa Γ) → next.

Euler geodezija (p,v) → next.

diskretna geodezija na putanji p_t.

Klizni prozor → p_t; g_ii=1/p_i; Γ^i_ii=−1/(2p_i).
Jedan Euler korak geodezije od (p, v):
  p' = p + η v
  v' = v − η Γ(p) ⊙ v²
  (projekcija nazad na simplex)

Skor iz p' excess vs global + smer v'; ban last; jedna next.
CSV ceo, seed=39.
"""



import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
WINDOW = 100
ETA = 1.0
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def window_p(draws: np.ndarray, end: int, w: int = WINDOW) -> np.ndarray:
    start = max(0, end - w)
    chunk = draws[start:end]
    cnt = Counter(chunk.reshape(-1).tolist())
    n_slots = max(len(chunk) * FRONT_SELECT, 1)
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def project_simplex(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def christoffel_diag(p: np.ndarray) -> np.ndarray:
    return -0.5 / np.clip(p, 1e-18, None)


def geodesic_step(p: np.ndarray, v: np.ndarray, eta: float = ETA) -> tuple[np.ndarray, np.ndarray]:
    """Jedan diskretni geodezijski korak (dijagonalna Fisher)."""
    gamma = christoffel_diag(p)
    p_new = project_simplex(p + eta * v)
    v_new = v - eta * gamma * (v ** 2)
    # tangent: sum v ~ 0 na simplexu
    v_new = v_new - v_new.mean()
    return p_new, v_new


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def number_scores(
    p_pred: np.ndarray,
    v_pred: np.ndarray,
    p_glob: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            # geodezijska predikcija mase + smer brzine
            out[n] = float((p_pred[i] - p_glob[i]) + 0.5 * v_pred[i])
    return out


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
    ban: set[int],
) -> float:
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws: np.ndarray, score: dict[int, float], ban: set[int]) -> list[int]:
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v5(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    n = len(draws)
    p0 = window_p(draws, n - 1, WINDOW)
    p1 = window_p(draws, n, WINDOW)
    v = p1 - p0
    v = v - v.mean()

    p_pred, v_pred = geodesic_step(p1, v, ETA)
    p_glob = global_p(draws)
    score = number_scores(p_pred, v_pred, p_glob, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {n} | seed={SEED} | WINDOW={WINDOW} | ETA={ETA} | ig_02_v5 geodesic")
    print(f"last: {last.tolist()}")
    print()

    print("=== geodezijski korak ===")
    print(
        {
            "v_l2": round(float(np.linalg.norm(v)), 6),
            "v_pred_l2": round(float(np.linalg.norm(v_pred)), 6),
            "dp_pred_l2": round(float(np.linalg.norm(p_pred - p1)), 6),
            "sum_p_pred": round(float(p_pred.sum()), 6),
        }
    )
    print()

    ranked = sorted(
        ((n_, float(score[n_])) for n_ in range(1, FRONT_N + 1) if n_ not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (p'−p_glob + 0.5 v', ban last) ===")
    print([(n_, round(sc, 6)) for n_, sc in ranked[:12]])
    print()

    combo = predict_next(draws, score, ban)
    print("=== next (ig_02_v5 geodesic) ===")
    print("next:", combo)
    print("overlap last:", sorted(set(combo) & ban))


if __name__ == "__main__":
    run_ig_02_v5()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | WINDOW=100 | ETA=1.0 | ig_02_v5 geodesic
last: [4, 5, 6, 11, 12, 18, 28]

=== geodezijski korak ===
{'v_l2': 0.004518, 'v_pred_l2': 0.004514, 'dp_pred_l2': 0.004518, 'sum_p_pred': 1.0}

=== top12 skor (p'−p_glob + 0.5 v', ban last) ===
[(1, 0.014142), (29, 0.009472), (14, 0.007906), (27, 0.007091), (16, 0.007015), (24, 0.006953), (8, 0.00617), (38, 0.004035), (34, 0.001654), (20, 0.001235), (9, 0.001147), (7, 0.001024)]

=== next (ig_02_v5 geodesic) ===
next: [7, 9, 19, 20, 24, 30, 31]
overlap last: []
"""
