# Experiment log

Ten days of experiments, 8 to 17 May 2026. Scores are NDCG@5. "Val (id split)" is the early validation that split on `srch_id`, and "Val (date split)" is the temporal split used from 14 May onwards. The two are not comparable with each other, and neither is directly comparable with the leaderboard.

## Scoreboard

| Version | Val (id split) | Val (date split) | Public LB | Date | What changed |
|---|---|---|---|---|---|
| v1 to v2 | ~0.39 | | | 8 to 9 May | Basic features and a single LightGBM ranker |
| v3 | 0.4105 | | 0.40816 | 9 May | 98 features, 3 model ensemble |
| v4 | 0.4103 | | | 10 May | Added a binary classifier and a random order only model, both failed |
| v5 | 0.4143 | | 0.41191 | 10 May | 138 features, including `prop_avg_position` |
| v6 | 0.4157 | | 0.41210 | 11 May | Label gain 5 to 1 instead of 31 to 1 |
| v7 | | 0.3972 | 0.41224 | 14 May | Date based validation and 9 new features |
| v8 | | 0.3968 | | 14 May | 6 deviation and competitor features, no gain |
| v9 | | 0.3978 | 0.41254 | 14 May | NDCG truncation at 12, back to the 138 features |
| Lean ensemble | | 0.3983 | 0.41348 | 14 May | 4 structurally different models |
| Depth ensemble | | 0.3996 | 0.41465 | 15 May | 6 models with deeper trees (500 leaves) |
| Random only ensemble | | 0.3820 | | 15 May | Trained on random order searches only, worse at every step |
| Seed diversity | | 0.3999 | 0.41469 | 15 May | Query weighting and seed averaging |
| **Improved** | | **0.3999** | **0.41599** | **15 May** | **Encoding leak fixed, RRF fusion, retrain at 1.25x rounds** |
| Wider ensemble | | 0.3999 | | 16 May | 15 models on the fixed features, worse than 3 |
| Optuna single | | 0.4004 | 0.41561 | 16 May | Best of 60 trials, worse on the leaderboard |
| Optuna ensemble | | 0.4016 | 0.41599 | 16 May | Tuned models with seed diversity, no leaderboard change |
| **Unbiased ensemble** | | **0.4023** | **0.41637** | **17 May** | **Position debiased rates and a price feature fix** |
| Two way blend | | | 0.41665 | 17 May | Unbiased 50 and Optuna 50, blending across feature sets |
| **Three way blend** | | | **0.41682** | **17 May** | **Unbiased 50, Optuna 25, Improved 25. Final submission** |
| Paper features | | 0.4033 | 0.41600 | 17 May | Count features and composites from the literature, worse on the leaderboard |

The final submission scored 0.41792 on the private leaderboard, 19th of 146 teams.

## Where the gains came from

| Change | Leaderboard gain |
|---|---|
| 98 to 138 features, including `prop_avg_position` | +0.00375 |
| Label gain 5 to 1 | +0.00019 |
| Fewer rounds, picked on the date split | +0.00014 |
| NDCG truncation at 12 | +0.00030 |
| 4 structurally different models | +0.00094 |
| Deeper trees and a wider ensemble | +0.00117 |
| **Encoding fix, RRF and 1.25x retrain** | **+0.00130** |
| Unbiased features alone | +0.00036 |
| Unbiased features inside the three way blend | +0.00081 |

## Notes on the main steps

**Feature breakthrough (10 May).** Going from 98 to 138 features was the largest single jump of the whole project. The standout was `prop_avg_position`, the average position Expedia gave a hotel in its own sorted results. It works as a summary of Expedia's internal quality signal, since hotels that Expedia keeps near the top tend to be good ones.

**Label gain (11 May).** LightGBM's default label gain treats a booking (relevance 5) as 31 times a click. Setting the gains to the raw relevance values, 5 and 1, gave a small improvement.

**The id is not time (14 May).** `srch_id` turned out to have no relationship with the search timestamp, so the original validation split was effectively random. Switching to a split on the timestamp lowered every validation score but made model choices more honest. Early stopping also picked fewer rounds, which helped slightly on the leaderboard.

**Ensembles (14 to 15 May).** Structural diversity (different depths and leaf counts) helped. Seed averaging alone barely moved anything. Training only on randomly ordered searches got worse at every step, since it throws away 70% of the data.

**Encoding leak (15 May).** Two bugs in the target encoding. The smoothing prior was computed once on all the training data, which leaked held out labels into the encoding of those same rows. Test rows were also encoded with tables fitted on all the training data, so they were less noisy than the out of fold training features the model had learned from. Moving the prior inside the fold loop and averaging the five fold tables for test fixed both. Together with RRF fusion and a longer final retrain, this was the biggest gain after the feature expansion.

**Hyperparameter search (16 May).** A 60 trial Optuna search found the best validation scores so far and did nothing on the leaderboard. From here on, validation gains from tuning alone stopped carrying over.

**Position debiasing (16 to 17 May).** Rates computed only from randomly ordered searches, plus features for the gap between biased and unbiased rates. Also fixed `price_vs_hist` and `price_discount` for hotels with no historical price, where `exp(0) = 1` had produced values around 200 and minus 200. This gave a new feature set, and blending models trained on it with models trained on the older features is what produced the final score.

**Blending (17 May).** Every earlier blend of models trained on the same features had rank correlations above 0.99 and gained nothing. Blends across feature sets sat around 0.98 and gave real improvements. Weighting the new feature set at 50% and the two older ensembles at 25% each gave the final submission.

**What did not help (17 May).** Count features, an `lr_score` feature and composite features inspired by a published paper on the original competition. Validation went up by 0.001 and the leaderboard went down slightly. In hindsight a 50/50 blend with the previous best would have been a safer bet than submitting the new model alone.

## Things that failed along the way

- A binary booking classifier added to the ranker ensemble
- Inverse propensity weighting for position bias (validation 0.4105 to 0.4035)
- Training only on randomly ordered searches
- Wider ensembles of 12 to 15 models on the same features
- Deviation and competitor features beyond the 138 feature set
- XGBoost rankers, which were competitive alone (0.4004 validation) but never got picked by the greedy selection
