# Changelog

## 2026-11-27

### Experiment matrix complete

- All 31 experiments completed
- 3 new const-LR swap experiments added (#28-30), confirming swap effect under paper-strict protocol
- g24 B/MDP/const/swap0.15 achieves test type1 = +1.76 (best paper-strict result)
- Swap effect confirmed not dependent on warmup_cosine LR scheduling
- README updated with complete 2×2×2 matrix and const-LR verification

## 2026-11-26

### Initial release

- Complete experiment matrix: 28+ controlled experiments
- g24 architecture × route × LR 3×2×2 matrix (12 experiments)
- dense124 route × LR 2×2 matrix (4 experiments, historical baseline)
- Intervention sweep: swap p∈{0.10,0.15,0.25,0.50} + noise + drop + gate-invar (7 experiments)
- Swap × route × architecture 2×2×2 matrix (7 of 8 complete + 3 const-LR verification)
- Baselines: author_release 12ep, pos_aug POC (2 experiments)
- Comprehensive README with synthetic-to-real gap analysis
- Reproducibility guide, environment specification, and audit trail

### Key findings

- Synthetic-to-real gap: 5+ dB median, with type1 val-test Spearman ρ = +0.12
- Only g24 B/MDP achieves positive test type1 without intervention (+1.29 to +1.63)
- Position-swap p=0.15 is the most effective anti-shortcut intervention (+2.30 dB on dense124)
- Swap effect confirmed on g24 architecture and under constant LR (paper-strict protocol)
- Route × architecture interaction: MDP helps g24B but hurts g24C
