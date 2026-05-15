# Worklist: Top-10 worst geometric violations

**Created:** 2026-05-15
**Updated:** 2026-05-16 — RESOLVED. Top-9 8-side trees fixed manual + rule relaxation (8-side max_dist 2→3 per visual validation RA). 0 remaining 8-side violations.
**Source:** `reports/audit_impossible_visibility/findings.csv` (severity=violation, sorted desc by n_sides_bunch)
**Strategy:** fix top-10 worst via manual tool, accept remaining 4-side violations sebagai GT noise floor.

Rule violation = bunch tidak punya geometric valid home (no candidate `home ∈ appearance_sides` di mana semua appearance lain dalam circular distance ≤ N/4).

Approach per tree: buka 4 (atau 8) gambar di tool annotator, identifikasi link salah yg connect offending side ke bunch utama, hapus link itu (atau split bunch jadi 2), save.

---

## 1. DAMIMAS_A21B_0824 bunch#1 (B1) — 8/8 sides ⚠️ PALING EKSTRIM

- Appearance: `sisi_1,sisi_2,sisi_3,sisi_4,sisi_5,sisi_6,sisi_7,sisi_8`
- Offending: `sisi_4, sisi_5, sisi_6` (opposite hemisphere)
- Links touching: `lnk-7, lnk-8, lnk-9, lnk-10, lnk-11, lnk-12, lnk-13`
- Action: bunch ke-merge dgn 1-2 bunch lain di sisi opposite. Split jadi 2-3 bunch terpisah (cluster s7-s8-s1-s2-s3 vs cluster s4-s5-s6).
- Images: `Brand-New-Dataset-YOLO/images/DAMIMAS_A21B_0824_{1..8}.jpg`

## 2. DAMIMAS_A21B_0812 bunch#1 (B1) — 7/8 sides

- Appearance: `s1,s2,s3,s4,s6,s7,s8` (skip s5)
- Offending: `sisi_4, sisi_6`
- Links: `lnk-0..lnk-5`
- Action: split jadi 2 bunch (s7-s8-s1-s2-s3 vs s4 vs s6 — atau merge s4 ke cluster1 + drop link s6).
- Images: `DAMIMAS_A21B_0812_{1..8}.jpg`

## 3. DAMIMAS_A21B_0812 bunch#2 (B3) — 7/8 sides

- Same tree as #2, diff bunch
- Appearance: `s1,s2,s3,s4,s6,s7,s8`
- Offending: `sisi_4, sisi_6`
- Links: `lnk-8..lnk-13`
- Action: similar split pattern.

## 4. DAMIMAS_A21B_0823 bunch#5 (B3) — 7/8 sides

- Appearance: `s1,s2,s3,s4,s5,s6,s7` (skip s8)
- Offending: `sisi_6, sisi_7`
- Links: `lnk-16..lnk-21`
- Action: cluster s1-s5 valid; s6-s7 mungkin bunch lain.
- Images: `DAMIMAS_A21B_0823_{1..8}.jpg`

## 5. DAMIMAS_A21B_0848 bunch#2 (B3) — 7/8 sides

- Appearance: `s1,s2,s3,s4,s5,s7,s8` (skip s6)
- Offending: `sisi_4, sisi_5`
- Links: `lnk-6..lnk-11`
- Action: split — cluster s7-s8-s1-s2-s3 vs s4-s5.
- Images: `DAMIMAS_A21B_0848_{1..8}.jpg`

## 6. DAMIMAS_A21B_0811 bunch#7 (B3) — 6/8 sides

- Appearance: `s1, s4, s5, s6, s7, s8` (skip s2, s3)
- Offending: `sisi_1` (cluster utama s4-s8)
- Links: `lnk-16, lnk-20, lnk-24, lnk-26, lnk-32`
- Action: drop link yg connect sisi_1 ke bunch — sisi_1 mungkin bunch terpisah.
- Images: `DAMIMAS_A21B_0811_{1..8}.jpg`

## 7. DAMIMAS_A21B_0812 bunch#3 (B2) — 6/8 sides

- Same tree as #2 dan #3, ketiga-bunch bermasalah di tree ini → kemungkinan annotator over-link sistematik di 0812
- Appearance: `s1, s2, s5, s6, s7, s8` (skip s3, s4)
- Offending: `sisi_2`
- Links: `lnk-14..lnk-18`
- Action: drop link s2 atau split.

## 8. DAMIMAS_A21B_0814 bunch#2 (B3) — 6/8 sides

- Appearance: `s1, s2, s3, s4, s7, s8` (skip s5, s6)
- Offending: `sisi_4`
- Links: `lnk-3..lnk-7`
- Action: drop link yg connect s4 ke bunch utama (s7-s8-s1-s2-s3).
- Images: `DAMIMAS_A21B_0814_{1..8}.jpg`

## 9. DAMIMAS_A21B_0815 bunch#8 (B3) — 6/8 sides

- Appearance: `s2, s3, s4, s5, s6, s7` (skip s1, s8)
- Offending: `sisi_7`
- Links: `lnk-17..lnk-21`
- Action: drop link s7 — cluster utama s2-s6.
- Images: `DAMIMAS_A21B_0815_{1..8}.jpg`

## 10. DAMIMAS_A21B_0817 bunch#3 (B1) — 6/8 sides

- Appearance: `s1, s2, s3, s4, s5, s6` (skip s7, s8)
- Offending: `sisi_6`
- Links: `lnk-6..lnk-10`
- Action: drop link s6 — cluster utama s1-s5.
- Images: `DAMIMAS_A21B_0817_{1..8}.jpg`

---

## Tracking checklist (FINAL — all 8-side cleared)

- [x] 1. 0824 bunch#1 (8/8) — manual fix
- [x] 2. 0812 bunch#1 (7/8) — manual fix
- [x] 3. 0812 bunch#2 (7/8) — manual fix
- [x] 4. 0823 bunch#5 (7/8) — manual fix
- [x] 5. 0848 bunch#2 (7/8) — manual fix + visually-validated 6/8
- [x] 6. 0811 bunch#7 (6/8) — auto-cleared by rule relaxation (max_dist 2→3)
- [x] 7. 0812 bunch#3 (6/8) — manual fix
- [x] 8. 0814 bunch#2 (6/8) — auto-cleared
- [x] 9. 0815 bunch#8 (6/8) — auto-cleared
- [x] 10. 0817 bunch#3 (6/8) — auto-cleared

**Final result (2026-05-16):** 62 → 42 violations (−20). 0 remaining 8-side. Tersisa 42 di 4-side trees (4/4 sides — rule unchanged).

## Notes

- Tree 0812 punya 3 bunch bermasalah (#1, #2, #3) → fix sekaligus saat buka tree ini.
- Cluster 0811-0848 → 6 dari 10 worst di sini. Suggest RA review siapa annotator session ini.
- Action interpretasi koord-only — visual inspection final say (per pengalaman 0335/0323/0362).
