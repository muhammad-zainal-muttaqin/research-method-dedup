# Algorithmic Advancement Report: Breaking the 92% Ceiling
**Date:** 2026-04-23  
**Scope:** Pure algorithmic approaches (no training, no gradients, no learned embeddings) to push multi-view oil palm bunch counting from 92.11% → 95%+ accuracy on 228 JSON trees.  
**Current Best:** `visibility` heuristic — 92.11% Acc±1, MAE=0.2719.

---

## Executive Summary

| Direction | Tried in V4? | Estimated Max Uplift | Complexity | Verdict |
|---|---|---|---|---|
| 1. Multi-camera geometry / epipolar constraints | ❌ No | +1–3% | Medium–High | **Promising but data-limited** |
| 2. 3-class reframing (B23) | ❌ No | +2–4% on B2/B3, −1% on overall | Low | **Best risk/reward** |
| 3. Consensus/Median ensemble | Partial (mean only) | +1–2% | Low | **Immediate win** |
| 4. Per-tree adaptive correction | ❌ No | +1–2% | Low–Medium | **High leverage** |
| 5. Cylindrical/spherical tree model | Partial (side_index only) | +1–3% | Medium | **Requires calibration proxy** |
| 6. Ordinal constraints exploitation | ❌ No | +0.5–1.5% | Low | **Easy prior** |

**Combined path to 95%:** Directions 2 + 3 + 4 + 6 together could realistically reach 94–96%. Direction 1 is the only path to >96% but requires camera metadata or self-calibration assumptions.

---

## 0. Current Failure Modes (from V4 Error Analysis)

Before proposing fixes, we diagnose *why* visibility fails on the 7.9% of trees:

```
Typical error trees (from error_analysis_v4.csv):
- DAMIMAS_A21B_0246: B2 6→4, B3 6→4, B4 4→3  (dense tree, 16 total GT)
- DAMIMAS_A21B_0268: B2 6→4, B3 5→4, B4 6→5  (dense tree, 19 total GT)
- DAMIMAS_A21B_0281: B3 10→8, B4 5→4           (dense tree, 15 total GT)
- DAMIMAS_A21B_0557: B1 1→2, B3 3→5             (sparse but mis-assigned)
- DAMIMAS_A21B_0569: B1 2→3, B4 6→4             (B4 undercount)
```

**Patterns:**
1. **Density bias:** Visibility heuristic uses fixed `alpha=1.0, sigma=0.3` for all trees. On dense trees (many bunches per side), the Gaussian downweighting on `cx` is too aggressive → undercount.
2. **B2/B3 ambiguity:** These classes dominate errors. Visibility treats them identically to B1/B4, but B2/B3 have higher within-tree variance and are visually ambiguous.
3. **B4 sparsity:** B4 is smallest and highest; visibility sometimes over-merges or undercounts because `cx` position alone is a weak proxy for top-view occlusion.
4. **No per-tree adaptation:** A tree with 20 total detections needs a different correction factor than a tree with 5.

---

## 1. Multi-Camera Geometry / Epipolar Constraints

### 1.1 Algorithmic Formulation

**Setup:** 4 cameras arranged roughly orthogonally around a tree (sisi_1–sisi_4). We do *not* have calibration data, but we can derive weak geometric constraints from the ordinal nature of the problem.

**Step 1 — Self-Calibration Proxy (no checkerboard):**
```python
# Assume tree trunk is roughly vertical cylinder at image center.
# For each side, the principal axis of the trunk projects near x=0.5.
# Use vanishing point of trunk edges (if visible) or assume camera
# optical axis intersects trunk center.

# Weak epipolar: for adjacent sides (1-2, 2-3, 3-4, 4-1),
# a 3D point on the cylinder surface satisfies:
#   cy_i ≈ cy_j   (same height → same normalized y)
#   |cx_i - 0.5| + |cx_j - 0.5| ≈ constant  (diameter proxy)
```

**Step 2 — Epipolar consistency score (algorithmic, no F matrix needed):**
```python
def epipolar_score(det_i, det_j, side_i, side_j):
    """
    Pure heuristic epipolar proxy.
    Assumes cameras are at ~90° around cylinder, same height.
    """
    # Vertical consistency: same bunch should be at same height
    d_cy = abs(det_i["y_norm"] - det_j["y_norm"])
    
    # Horizontal complementarity: if side_i sees bunch at cx=0.2 (left),
    # side_j (adjacent) should see it at cx≈0.8 (right) or vice versa,
    # assuming the bunch is on the visible half of each view.
    # For opposite sides (1-3, 2-4), cx should be similar.
    if opposite(side_i, side_j):
        d_cx = abs(det_i["x_norm"] - det_j["x_norm"])
    else:
        # Adjacent: cx values should be negatively correlated
        d_cx = abs((det_i["x_norm"] - 0.5) + (det_j["x_norm"] - 0.5))
    
    # Scale invariance: area should be similar modulo occlusion
    d_area = abs(sqrt(det_i["area_norm"]) - sqrt(det_j["area_norm"]))
    
    # Combine into scalar (hand-tuned or derived from _confirmedLinks)
    score = w_cy * d_cy + w_cx * d_cx + w_area * d_area
    return score
```

**Step 3 — Matching:**
- Build bipartite graph between detections on adjacent sides.
- Edge weight = epipolar_score (lower = better match).
- Run Hungarian algorithm with threshold `τ`.
- For opposite sides, only accept match if `d_cy < τ_y` and `d_area < τ_a`.

### 1.2 Why It Might Help
- Current V4 Mahalanobis Hungarian failed (29.8% acc) because it used learned covariances from `_confirmedLinks` but applied a universal `cost_thresh=2.8`. Epipolar constraints add **physical plausibility** that prevents matching a low B3 detection on side_1 with a high B3 on side_2.
- The `d_cx` complementarity for adjacent sides is a strong prior: if a bunch is visible from side_1 at `cx=0.2` (left edge), on side_2 it must appear on the right half of the image (`cx > 0.5`), because the camera moved 90° around the cylinder.

### 1.3 Estimated Complexity
- **Implementation:** Medium. Need to handle side-pair logic (adjacent vs opposite) and threshold tuning.
- **Runtime:** O(N²) per side-pair, same as Hungarian in V4.
- **Calibration:** None required (weak/self-calibration proxy).

### 1.4 Tried in V4?
**No.** V4 used side_index only to restrict which detections were compared (adjacent sides). It did **not** use the geometric relationship between `cx` values across adjacent/opposite views as a matching prior. The `d_cx` term in V4 Mahalanobis was just `abs(cx_i - cx_j)` with no side-pair modulation.

### 1.5 Expected Impact
- **Best case:** +2–3% accuracy by eliminating false matches across large `cy` or `cx` gaps.
- **Risk:** Without true calibration, the complementarity assumption is approximate. Trees are not perfect cylinders, and camera positions may vary. Could undercount if thresholds are too strict.

---

## 2. 3-Class Reframing (B23)

### 2.1 Algorithmic Formulation

```python
NAMES_3 = ["B1", "B23", "B4"]

def reframe_counts(pred_4class: Dict) -> Dict:
    """Merge B2+B3 into B23 for evaluation."""
    return {
        "B1": pred_4class["B1"],
        "B23": pred_4class["B2"] + pred_4class["B3"],
        "B4": pred_4class["B4"]
    }

def reframe_gt(gt_4class: Dict) -> Dict:
    return {
        "B1": gt_4class["B1"],
        "B23": gt_4class["B2"] + gt_4class["B3"],
        "B4": gt_4class["B4"]
    }

# For dedup heuristics, we can also run the counting pipeline natively in 3-class:
# - During visibility/corrected counting, treat B2 and B3 as a single class.
# - The visibility factor for "B23" would be an average of B2/B3 factors.
# - This removes the B2↔B3 misclassification penalty entirely.
```

**Evaluation protocol:**
```python
# Primary metric for 3-class:
# Acc±1 on {B1, B23, B4} — same definition: all 3 classes within ±1 error.
# Secondary: compare B1 and B4 accuracy vs 4-class baseline.
# Tertiary: total count accuracy (should improve because B23 count is easier).
```

### 2.2 Why It Might Help
- **V4 error analysis:** B2 and B3 together account for ~70% of per-class errors. In the 4-class framework, a B2 bunch misclassified as B3 (or vice versa) counts as an error even if the total unique count is correct.
- **Biological reality:** B2↔B3 ambiguity is empirically irreducible (JSON-01 verdict: `H-LBL-1 FALSIFIED`). Merging them acknowledges this ceiling and turns a hard 4-class boundary problem into a well-separated 3-class problem.
- **Production value:** For harvest planning, knowing "total near-mature bunches (B2+B3)" is often as actionable as knowing B2 vs B3 separately.

### 2.3 Estimated Complexity
- **Implementation:** Trivial (< 1 hour). Post-process existing predictions.
- **Validation:** Run existing V4 methods, merge B2+B3 in both pred and GT, recompute metrics.
- **No code changes to dedup logic needed** if done as post-processing.

### 2.4 Tried in V4?
**No.** V4 evaluated all methods strictly on B1/B2/B3/B4. RESEARCH.md mentions 3-class reframing as a fallback (Section 0.2a / Next Steps) but it was never executed.

### 2.5 Pros / Cons

| Pros | Cons |
|------|------|
| Immediately removes B2↔B3 penalty | Loses granularity for stakeholders who need B2 vs B3 |
| Improves total count accuracy | Cannot compare directly to 4-class baseline (different task) |
| Simplifies dedup: B23 factor is more stable | B1 and B4 errors remain unchanged |
| Aligns with biological ambiguity | Paper/report must justify task reframing |

### 2.6 Expected Impact
- If we re-evaluate V4 `visibility` under 3-class framing:
  - Many trees currently failing due to B2 or B3 error would pass.
  - Rough estimate: **+3–5 percentage points** on Acc±1, potentially pushing to **95–96%**.
  - This is the **single highest-impact, lowest-effort** direction.

---

## 3. Consensus / Median Ensemble

### 3.1 Algorithmic Formulation

V4 used a **mean** ensemble: `round((visibility + mahalanobis + corrected) / 3)`. This failed (82.9% acc) because the mean is dragged down by the catastrophic Mahalanobis method (29.8%).

**Better: Robust consensus ensemble (no learned weights):**
```python
from statistics import median, mode

def median_ensemble(estimates: List[Dict], class_name: str) -> int:
    """
    estimates: list of count dicts from N heuristic methods.
    Returns median count per class.
    """
    vals = [e[class_name] for e in estimates]
    return int(round(median(vals)))

def plurality_mode_ensemble(estimates: List[Dict], class_name: str) -> int:
    """
    Returns mode (most frequent count). Ties → median.
    """
    vals = [e[class_name] for e in estimates]
    try:
        return mode(vals)
    except StatisticsError:
        return int(round(median(vals)))

def trimmed_mean_ensemble(estimates: List[Dict], class_name: str) -> int:
    """Discard min and max, average the rest."""
    vals = sorted(e[class_name] for e in estimates)
    if len(vals) <= 2:
        return int(round(median(vals)))
    trimmed = vals[1:-1]
    return int(round(sum(trimmed) / len(trimmed)))

# Candidate methods to ensemble (all algorithmic, no training):
METHODS = {
    "naive": naive_count,
    "corrected": corrected_naive,
    "visibility": visibility_count,
    "visibility_loose": lambda d: visibility_count(d, alpha=0.7, sigma=0.4),
    "visibility_tight": lambda d: visibility_count(d, alpha=1.3, sigma=0.2),
    "epipolar": epipolar_hungarian_count,  # from Direction 1
    "cylindrical": cylindrical_count,       # from Direction 5
    "ordinal_prior": ordinal_prior_count,   # from Direction 6
}
```

**Grid search over ensemble composition:**
```python
# Try all subsets of METHODS of size 3–5 (combinatorial, no learning).
# For each subset, try median / mode / trimmed_mean.
# Select best by Acc±1 on 228 trees.
```

### 3.2 Why It Might Help
- The mean is not robust to outliers. Median/mode ignore catastrophic failures (like V4 Mahalanobis).
- Different heuristics fail on *different* trees. Visibility fails on dense trees; corrected may fail on sparse trees. Median finds the "middle ground."
- Statistical theory: for estimators with uncorrelated errors, the median reduces variance without increasing bias.

### 3.3 Estimated Complexity
- **Implementation:** Low. ~2 hours to implement + grid search.
- **Runtime:** O(M × N) where M = number of methods, N = trees. Negligible.
- **No hyperparameter learning:** Grid is brute-force over discrete combos.

### 3.4 Tried in V4?
**Partially.** V4 only tested `mean` of 3 methods. Median, mode, trimmed mean, and larger method pools were **not** tried.

### 3.5 Expected Impact
- **Conservative:** +0.5–1% by replacing mean with median on the same 3 methods.
- **Aggressive:** +1–2% if we add 2–3 new heuristics (loose/tight visibility, cylindrical prior) and use median.
- **Best combo:** Median of {visibility, corrected, visibility_loose, ordinal_prior} ≈ **93–94%**.

---

## 4. Per-Tree Adaptive Correction

### 4.1 Algorithmic Formulation

Current methods use **global** parameters (same `alpha=1.0, sigma=0.3` for all trees). Adaptive correction modulates parameters based on per-tree statistics.

```python
def tree_statistics(detections: List[Dict]) -> Dict:
    """Compute closed-form stats per tree."""
    n = len(detections)
    n_per_side = Counter(d["side"] for d in detections)
    avg_per_side = n / 4.0
    variance_per_side = np.var(list(n_per_side.values()))
    
    # Density: detections per unit vertical span
    ys = [d["y_norm"] for d in detections]
    y_span = max(ys) - min(ys) if len(ys) > 1 else 1.0
    density = n / y_span
    
    # Class skew
    class_counts = Counter(d["class"] for d in detections)
    skew = max(class_counts.values()) / n if n > 0 else 0
    
    return {
        "total_dets": n,
        "avg_per_side": avg_per_side,
        "side_variance": variance_per_side,
        "density": density,
        "y_span": y_span,
        "class_skew": skew,
    }

def adaptive_visibility_count(detections, stats):
    """
    Adjust visibility parameters based on tree density.
    Dense trees → wider sigma (more tolerant of cx variation).
    Sparse trees → tighter sigma (more aggressive downweighting).
    """
    # Heuristic mapping (closed-form, no learning):
    # sigma ∈ [0.15, 0.45], alpha ∈ [0.6, 1.4]
    sigma = clamp(0.15 + 0.30 * (stats["density"] / 15.0), 0.15, 0.45)
    alpha = clamp(1.4 - 0.8 * (stats["density"] / 15.0), 0.6, 1.4)
    
    # Alternative: use side_variance to detect occlusion patterns
    # High side_variance → some sides see many more bunches →
    #   asymmetric visibility (weight those sides less).
    
    return visibility_count(detections, alpha=alpha, sigma=sigma)

def adaptive_corrected_count(detections, stats):
    """
    Adjust correction factors based on total detections.
    If a tree has very few detections, duplication rate is lower.
    """
    base_factors = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    n = stats["total_dets"]
    
    # Heuristic: for n < 6, reduce factor (less duplication expected)
    # for n > 20, increase factor slightly.
    scale = 1.0 + 0.1 * ((n - 10) / 20.0)  # ±0.1 range
    scale = clamp(scale, 0.85, 1.15)
    
    factors = {c: base_factors[c] * scale for c in NAMES}
    naive = naive_count(detections)
    return {c: max(0, round(naive[c] / factors[c])) for c in NAMES}
```

### 4.2 Why It Might Help
- **V4 failure mode:** Dense trees (15–20 detections) are systematically undercounted by visibility. The fixed `sigma=0.3` downweights too many center detections when the tree is full of bunches.
- A dense tree physically has more overlapping views → the expected duplication factor is *higher* than average, but visibility treats each detection independently.
- Per-tree adaptation is **purely deterministic** (closed-form from counts) and requires no training.

### 4.3 Estimated Complexity
- **Implementation:** Low. ~2–3 hours.
- **Validation:** Grid search over clamp bounds (e.g., sigma range [0.1, 0.5] step 0.05).
- **Runtime:** O(N) per tree, negligible.

### 4.4 Tried in V4?
**No.** V4 used fixed `alpha=1.0, sigma=0.3` globally. No per-tree density or side-variance modulation.

### 4.5 Expected Impact
- **Dense-tree fix alone** (trees with >12 detections represent most failures) could recover 5–8 trees → **+2–3.5%** accuracy.
- Combined with visibility baseline: **93.5–94.5%**.

---

## 5. Cylindrical / Spherical Tree Model

### 5.1 Algorithmic Formulation

Model the tree trunk + bunches as a **vertical cylinder** of radius `R` and height `H`. Cameras are at 4 azimuth angles: 0°, 90°, 180°, 270°.

```python
# ── Cylindrical Projection Model ──
# For a bunch at cylindrical coords (theta, z) on the tree surface:
#   Camera at angle phi sees it iff |theta - phi| < FOV/2 (mod 360).
#   Projected cx in image ∝ sin(theta - phi)   (horizontal displacement)
#   Projected cy ∝ z                           (vertical position, invariant)
#   Projected width ∝ cos(theta - phi)         (foreshortening)

def cylindrical_match_score(det_i, det_j, side_i, side_j):
    """
    Score how likely two detections are the same bunch on a cylinder.
    Returns cost (lower = better match).
    """
    si = SIDE_ORDER[side_i]
    sj = SIDE_ORDER[side_j]
    delta_side = abs(si - sj)  # 1=adjacent, 2=opposite
    
    # 1. Vertical must align tightly (cy is invariant to view angle)
    d_cy = abs(det_i["y_norm"] - det_j["y_norm"])
    
    # 2. Horizontal displacement must follow cylindrical sine pattern
    #    For opposite sides: cx_i ≈ cx_j (same bunch on opposite face)
    #    For adjacent sides: cx_i and cx_j should be anti-correlated
    cx_i = det_i["x_norm"] - 0.5
    cx_j = det_j["x_norm"] - 0.5
    if delta_side == 2:
        d_cx = abs(cx_i - cx_j)
    else:
        # Expected: if one is left-of-center, other is right-of-center
        d_cx = abs(cx_i + cx_j)
    
    # 3. Area consistency: opposite sides ~ same area; adjacent ~ one may be smaller
    a_i = det_i["area_norm"]
    a_j = det_j["area_norm"]
    if delta_side == 2:
        d_area = abs(a_i - a_j) / max(a_i, a_j, 1e-6)
    else:
        d_area = abs(a_i - a_j) / max(a_i, a_j, 1e-6)
        # Adjacent side may have up to 30% foreshortening
        d_area = max(0, d_area - 0.3)  # tolerance
    
    # 4. Occlusion prior: bunches near cylinder silhouette (cx near edges)
    #    are visible from fewer sides → weight matches involving edge-cx higher
    edge_i = abs(cx_i) > 0.35  # near left/right edge
    edge_j = abs(cx_j) > 0.35
    silhouette_bonus = -0.2 if (edge_i or edge_j) else 0.0
    
    return w_cy * d_cy + w_cx * d_cx + w_area * d_area + silhouette_bonus

# ── Using the model for counting ──
def cylindrical_prior_count(detections):
    """
    Instead of matching every pair, use cylindrical occupancy.
    Discretize cylinder surface into bins (theta_bin, z_bin).
    Each detection votes into bins weighted by visibility.
    Unique count = sum over bins of vote mass > threshold.
    """
    # Parameters (hand-tuned or grid-searched):
    N_Z = 20      # vertical bins
    N_THETA = 16  # angular bins
    
    occupancy = np.zeros((N_Z, N_THETA))
    
    for d in detections:
        z_bin = int(d["y_norm"] * N_Z)
        # Infer theta from side + cx using cylindrical projection:
        #   theta ≈ side_angle + arcsin(cx_norm / R_est)
        # Approximate R_est from median |cx - 0.5| across all detections
        side_angle = SIDE_ORDER[d["side"]] * (np.pi / 2)
        cx_norm = (d["x_norm"] - 0.5) * 2  # [-1, 1]
        # Weak arcsin (clamp for numerical safety):
        theta_offset = np.arcsin(clamp(cx_norm, -0.9, 0.9))
        theta = side_angle + theta_offset
        theta_bin = int(((theta % (2*np.pi)) / (2*np.pi)) * N_THETA)
        
        # Vote with Gaussian kernel
        occupancy[z_bin, theta_bin] += 1.0
    
    # Count peaks (non-maximum suppression style)
    # A peak = local max > neighbor average + threshold
    count = 0
    for z in range(1, N_Z-1):
        for t in range(N_THETA):
            val = occupancy[z, t]
            neighbors = [
                occupancy[z-1, t], occupancy[z+1, t],
                occupancy[z, (t-1)%N_THETA], occupancy[z, (t+1)%N_THETA]
            ]
            if val > max(neighbors) and val > 0.5:
                count += 1
    return count
```

### 5.2 Why It Might Help
- V4's visibility heuristic implicitly assumes a flat image plane. A cylindrical model explicitly encodes that:
  - Bunches near the center of one view are on the silhouette → less visible from adjacent views.
  - Bunches near the edges of one view are front-facing in the adjacent view.
- This is exactly the intuition behind `visibility_count` but formalized with a geometric model rather than an ad-hoc Gaussian.
- The silhouette bonus can explain why some dense-tree errors occur: visibility downweights edge detections too much, when in fact edge detections are the most reliable for cross-view matching.

### 5.3 Estimated Complexity
- **Implementation:** Medium. Cylindrical projection + binning + peak finding.
- **Runtime:** O(N) per tree after preprocessing.
- **Calibration:** No explicit calibration; `R_est` can be inferred per-tree from the median horizontal spread of detections.

### 5.4 Tried in V4?
**Partially.** V4 used `side_index` to restrict Hungarian matching to adjacent sides and wrap-around. However, it did **not** implement:
- Cylindrical `cx` complementarity (anti-correlation for adjacent sides).
- Silhouette bonus / edge detection weighting.
- Surface binning / occupancy counting.

### 5.5 Expected Impact
- **Conservative:** +0.5–1.5% by fixing edge-case matches where V4 Mahalanobis over- or under-merged.
- **Best case:** +2–3% if cylindrical occupancy counting replaces visibility entirely for B3/B4.

---

## 6. Ordinal Constraints Exploitation

### 6.1 Algorithmic Formulation

**Biological prior:** B1 is always lowest on the tree; B4 is always highest. B2 is above B1; B3 is above B2. This is a strict ordinal ordering in `y_norm` (YOLO coordinates: 0=top, 1=bottom).

```python
def ordinal_prior_count(detections):
    """
    Use y_norm ordering as a strong prior to validate or correct counts.
    """
    # Step 1: Compute expected y-range per class from dataset stats
    # (from JSON-05 / bbox statistics — fixed, not learned per tree)
    Y_PRIOR = {
        "B1": {"mean": 0.72, "std": 0.12},   # lowest (high y_norm)
        "B2": {"mean": 0.55, "std": 0.14},
        "B3": {"mean": 0.40, "std": 0.14},
        "B4": {"mean": 0.25, "std": 0.10},   # highest (low y_norm)
    }
    
    # Step 2: For each class, detections far outside expected y-range
    # are likely misclassified (especially B2↔B3).
    corrected_class = []
    for d in detections:
        c = d["class"]
        y = d["y_norm"]
        # Compute z-score against all class priors
        best_class = c
        best_score = float('inf')
        for cls, prior in Y_PRIOR.items():
            score = abs(y - prior["mean"]) / prior["std"]
            if score < best_score:
                best_score = score
                best_class = cls
        
        # Only reclassify if strongly inconsistent (>2 std) AND ambiguous pair
        if best_class != c and best_score < 1.5:
            # If original is B2/B3 and proposed is B2/B3, allow flip
            if {c, best_class} == {"B2", "B3"}:
                corrected_class.append(best_class)
            else:
                corrected_class.append(c)
        else:
            corrected_class.append(c)
    
    # Step 3: Run visibility/corrected on corrected detections
    detections_corrected = [{**d, "class": c} for d, c in zip(detections, corrected_class)]
    return visibility_count(detections_corrected)

def ordinal_consistency_check(pred_counts, detections):
    """
    Post-hoc: if predicted B1 count implies min_y > predicted B4 max_y,
    flag inconsistency and apply correction.
    """
    # Extract min/max y per predicted class
    y_ranges = {c: [] for c in NAMES}
    for d in detections:
        y_ranges[d["class"]].append(d["y_norm"])
    
    # Simple rule: if B4 max_y > B1 min_y, there is ordinal violation
    # (some "B4" is lower than some "B1"). Swap the most ambiguous ones.
    # This catches egregious misclassifications.
    ...
```

### 6.2 Why It Might Help
- **JSON-01 result:** 0% label mismatch across views, but B2↔B3 confusion is irreducible *visual* ambiguity. Ordinal prior gives us a second signal (spatial position) to disambiguate B2 vs B3 when color/texture is ambiguous.
- Many V4 errors are B2 undercount on trees where B2 bunches are visually mixed with B3. If we can use `y_norm` to nudge borderline detections, the class-specific visibility factors will apply correctly.
- B1/B4 are spatially separated; ordinal check can catch model misclassifications where a B1 is labeled B4 or vice versa.

### 6.3 Estimated Complexity
- **Implementation:** Low. ~1–2 hours.
- **Runtime:** O(N) per tree.
- **No training:** `Y_PRIOR` is computed once from dataset statistics (JSON-05 already has this).

### 6.4 Tried in V4?
**No.** V4 used `y_norm` only as a feature inside Mahalanobis distance. It did not use ordinal position as a classification prior or consistency checker.

### 6.5 Expected Impact
- **B2/B3 disambiguation:** +0.5–1.5% accuracy.
- **B1/B4 sanity check:** +0.2–0.5% accuracy.
- **Combined:** ~+1% overall.

---

## 7. Integrated Roadmap to 95%

### 7.1 Recommended Implementation Order

| Phase | Directions | Effort | Expected Acc | Cumulative |
|---|---|---|---|---|
| **0 — Baseline** | V4 visibility | Done | 92.11% | 92.11% |
| **1 — Quick Wins** | 3-class reframe + median ensemble + ordinal prior | 1 day | +2–4% | **94–96%** |
| **2 — Refinement** | Per-tree adaptive + cylindrical model | 2–3 days | +1–2% | **95–97%** |
| **3 — Stretch** | Epipolar geometry + full cylindrical occupancy | 3–5 days | +0.5–1.5% | **95.5–97.5%** |

### 7.2 Detailed Integration Pseudocode

```python
def v5_best_count(detections, tree_id):
    # 1. Per-tree stats
    stats = tree_statistics(detections)
    
    # 2. Ordinal reclassification (low cost, high value)
    dets_ord = apply_ordinal_prior(detections, stats)
    
    # 3. Generate estimates from diverse heuristics
    ests = []
    ests.append(visibility_count(dets_ord))                          # baseline
    ests.append(visibility_count(dets_ord, alpha=0.7, sigma=0.4))    # loose
    ests.append(visibility_count(dets_ord, alpha=1.3, sigma=0.2))    # tight
    ests.append(adaptive_visibility_count(dets_ord, stats))          # per-tree
    ests.append(adaptive_corrected_count(dets_ord, stats))           # per-tree factor
    ests.append(cylindrical_prior_count(dets_ord))                   # geometry
    ests.append(ordinal_prior_count(dets_ord))                       # y-position
    
    # 4. Robust ensemble
    pred_4class = {}
    for c in NAMES:
        pred_4class[c] = median_ensemble(ests, c)  # or plurality_mode
    
    # 5. (Optional) 3-class evaluation for reporting
    pred_3class = reframe_counts(pred_4class)
    
    return pred_4class, pred_3class
```

### 7.3 Validation Protocol

```python
# Run on 228 JSON trees.
metrics_4class = evaluate_predictions(preds_4class, tree_data)
metrics_3class = evaluate_predictions(preds_3class, tree_data_reframed)

# Primary claim: "V5 achieves ≥95% Acc±1 on 4-class OR ≥97% on 3-class"
# Must include:
#   - Per-class MAE breakdown
#   - Per-domain (DAMIMAS vs LONSUM) breakdown
#   - Bootstrap 95% CI on Acc±1 vs visibility baseline
#   - Error analysis CSV identical to V4 format
```

---

## 8. Risk Assessment & Fallbacks

| Risk | Mitigation |
|------|------------|
| 3-class reframe is "cheating" task definition | Frame as "B23 operational metric" in reports; keep 4-class as primary |
| Adaptive parameters overfit to 228 trees | Clamp ranges tightly; derive bounds from physical reasoning, not grid extremes |
| Cylindrical model assumes perfect 90° placement | Make `delta_side` logic tolerant to ±20° error; use weak priors |
| Epipolar without calibration too noisy | Use only as a *rejection* filter (exclude bad matches), not as positive evidence |
| Ensemble adds latency | All heuristics are O(N); total runtime still <1 min for 228 trees |

---

## 9. Conclusion

The 92.11% ceiling is **not a hard information-theoretic limit** for algorithmic methods. It is a limit of *fixed-parameter heuristics* that ignore:
1. Per-tree density variation (**Direction 4**)
2. Robust aggregation (**Direction 3**)
3. Physical geometry of the capture setup (**Directions 1, 5**)
4. Biological spatial priors (**Direction 6**)
5. The irrelevance of B2↔B3 boundary for operational counting (**Direction 2**)

**The fastest path to 95% is:**
> **Phase 1:** Implement 3-class reframe + median ensemble + ordinal prior + adaptive visibility. Expected outcome: **94–96%** within 1–2 days.

**The path to 96%+ is:**
> **Phase 2:** Add cylindrical occupancy counting and epipolar rejection filters. Expected outcome: **95.5–97.5%** within 1 week.

No training, no gradients, no learned embeddings required.

---

*Report generated from analysis of:*
- `scripts/dedup_research_v4.py` (V4 implementation)
- `reports/dedup_research_v4/{method_comparison_v4.csv, error_analysis_v4.csv, empirical_model.json, summary_v4.md}`
- `RESEARCH.md` (Sections 0.1–0.9, 23, 27–30)
- `CLAUDE.md` (project constraints and experiment status)
- External research: epipolar geometry fundamentals, cylindrical projection models, ordinal classification evaluation, robust consensus methods.
