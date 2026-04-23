Failing trees (v5 best, 14/228):
  - test split dominates: 6/14 (test only has 36 trees → 16.7% fail rate vs train 3.5%)
  - Primary error class: B3 (err=2 in 7 trees), then B4 (err=2 in 4 trees), B2 (err=2 in 4 trees)   
  - B1 only errs by 1 anywhere
  - All failures are err=2 on exactly one or two classes — not catastrophic

  Key insight: test split fails 19.4% (7 of 36) vs train 3.5% (7 of 192). Val 0% (11 trees). This is   critical. Suggests:
  - Current density factors tuned on full dataset → mild overfitting to train distribution
  - Or test trees systematically denser/different

  Let me check distribution of failing trees' total_gt.

  Failing totals: 7, 12, 13, 12, 7, 9, 11, 15, 16, 14, 9, 10, 14, 5. Median ≈ 11.5. Dataset median  
  similar. Not obviously density-driven.

  Now answer user thoughtfully. No code yet — they asked for research first.

  Diagnosis — 93.86% ceiling, 14 failing trees

  Error profile (v5 adaptive_corrected):
  - B3 errors dominate (err=2 on 7/14 trees), then B4 (4/14), B2 (4/14). B1 almost perfect.
  - Split imbalance: test fails 19.4% (7/36) vs train 3.5% (7/192), val 0%. This is the biggest red 
  flag — current density factors are tuned on aggregate stats dominated by train → mild distribution   shift on test.
  - All failures are off-by-2 on one/two classes. No catastrophic misses.
  - v6 shows 7 methods tied at 93.86%, different MAEs — ensemble with them can't break tie since    
  they agree on hard trees.

  Why tuning further is a trap:
  - Grid search further on 228 trees = benchmaxxing. Tuned hyperparameters become artifacts of this 
  split.
  - v6 tried 19 variants, all plateau → the corrections themselves are saturated.

  Proposed Research Plan (generalization-first, no benchmaxxing)

  Governance: Split protocol. Hyperparameters learned/chosen on train+val (204 trees) only. Test (36   trees) held out — evaluated once at the end per method. No grid search on test. Report with      
  bootstrap CI + leave-one-tree-out (LOTO) cross-validation to detect overfitting.

  Directions (ranked by expected generalization):

  1. LOTO cross-validation framework (prerequisite). Before any new method, refit BASE_FACTORS +    
  density slope per-fold (LOTO) so reported accuracy reflects generalization not fit. If current    
  93.86% drops to ~92% under LOTO, we know the ceiling is partly overfit and honest improvement     
  matters more than headline number.
  2. Physically-motivated per-tree density, not tree-count. Current _tree_density_scale uses n_total   linearly — crude. Replace with vertical stacking density (count of detections per unit y-span per   class). Bunches that stack vertically duplicate more (visible from multiple sides). Closed-form, 
  interpretable, no new parameters beyond one per class derived from dataset medians.
  3. Side-coverage consistency floor/ceiling (strong, underused). For class c: unique count must be 
  ≥ max per side AND ≤ naive/min-duplication. v6 floor_corrected uses this but only as floor. Make  
  it a bracket: if adaptive_corrected falls outside, snap to nearest bracket boundary. This is a    
  93.86% drops to ~92% under LOTO, we know the ceiling is partly overfit and honest improvement     
  matters more than headline number.
  2. Physically-motivated per-tree density, not tree-count. Current _tree_density_scale uses n_total   linearly — crude. Replace with vertical stacking density (count of detections per unit y-span per   class). Bunches that stack vertically duplicate more (visible from multiple sides). Closed-form, 
  interpretable, no new parameters beyond one per class derived from dataset medians.
  3. Side-coverage consistency floor/ceiling (strong, underused). For class c: unique count must be 
  ≥ max per side AND ≤ naive/min-duplication. v6 floor_corrected uses this but only as floor. Make  
  it a bracket: if adaptive_corrected falls outside, snap to nearest bracket boundary. This is a    
  pure constraint, impossible to overfit.
  4. Ordinal position consistency as soft constraint (not reclassification). Don't flip B2↔B3       
  classes — too risky. Instead: if predicted B3 detections are in B2-typical y-range, reduce the B3 
  divisor slightly (they're less likely duplicates because they're atypical). Pure modulation,      
  deterministic.
  5. B3 residual-specific correction. B3 drives 50% of failures. B3 has highest count variance (most   bunches/tree). Specifically: when naive B3 > 8, current factor 1.795 may under-correct very-dense   cases. Fit class-wise quadratic (not linear) density scaling on train+val, validate on test. Only   1 extra param.
  6. 3-class fallback ONLY as secondary report. User said "must stay 4-class" — keep as primary. But   report B23-merged as secondary diagnostic: shows how much of the gap is genuine B2/B3 irreducible   ambiguity vs algorithmic.

  Methods to deprecate / not pursue:
  - Cylindrical/epipolar on non-JSON — V5 already removed, data geometry not cylindrical.
  - More grid-search variants on existing heuristics — plateau confirmed.
  - Mahalanobis/Hungarian — failed in v4 (29.8%).

  Success criteria (stricter than before):
  - Train+val → test generalization gap < 3pp
  - LOTO Acc±1 ≥ point estimate − 2pp
  - Reduce mean-error-per-failing-tree from 2.93 → <2.5 even if Acc±1 doesn't budge (error-reduction   objective)

  Next step

  Want me to:
  - (a) implement the LOTO harness + physically-motivated density first (diagnostic — tells us if   
  93.86% is real), or
  - (b) sketch all 5 directions in dedup_research_v7.py and run the full comparison, or
  - (c) write a contract-work/v7-plan.md detailing algorithms + generalization protocol before any  
  code?

  I lean (a) — diagnostic before offense. Without LOTO, any v7 gain is untrustworthy. Once we know  
  the honest baseline we can attack the right gap.