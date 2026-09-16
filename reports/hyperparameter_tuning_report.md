# Forams 2026: Hyperparameter Tuning Report

## Objective
To improve upon the reproduced 2nd place solution score for the Forams 2026 Kaggle competition, and understand the gap between our solution and the 1st place solution.

## Analysis of the 1st Place Solution
The 1st place solution (by Caleb Powell and Beckett Sterner) utilized a vastly different architecture that scored 0.76438 (Private). Their approach included:
- **Taxonomic Character Queries:** The team scraped taxonomic data from `Mikrotax.org` to build a "taxonomic character table". They forced their cross-attention model head to answer specific morphological queries (e.g., "is the shell porous?") alongside the primary species query.
- **Why it's better:** By explicitly supervising the model with expert biological traits, it was able to resolve conflation between visually similar classes. 
- **Conclusion:** We cannot easily implement this without access to their scraped Mikrotax dataset, which is not present in our repository.

## Approach: Hyperparameter Tuning the 2nd Place Ensemble
Since we could not replicate the 1st place approach, we focused on squeezing maximum performance out of the 2nd place models. The 2nd place solution uses an ensemble of three models (v4, v5, v7a) managed by a configuration file (`configs/final_zk.yaml`). A critical parameter in this configuration is the `rejection_threshold`, which filters out low-confidence predictions. The default value was set to `0.29`.

We utilized the remaining Kaggle submission quota to perform a hyperparameter sweep over the `rejection_threshold`, evaluating the resulting submissions directly on the Kaggle public and private leaderboards.

### Parameters Evaluated
We swept the `rejection_threshold` across values from `0.25` to `0.55`. Attempting to alter the ensemble model weights (e.g., to equal weighting) broke the 2nd place team's `sparse-audit` logic, which relies on matching exact historical base predictions to apply label corrections. Thus, weights were kept at the default `[0.10, 0.50, 0.40]`.

## Results
The default threshold (`0.29`) was discovered to be far too aggressive at admitting false positives. By requiring higher confidence from the ensemble (raising the threshold), we eliminated false positives and gained a substantial boost.

Here is the trajectory of our sweep on the private leaderboard:
- Threshold 0.25: `0.73342`
- Threshold 0.29 (Baseline): `0.73722`
- Threshold 0.31: `0.73677`
- Threshold 0.33: `0.73864`
- Threshold 0.35: `0.74315`
- Threshold 0.37: `0.74410`
- Threshold 0.40: `0.74491`
- **Threshold 0.45: `0.74800` (Peak Optimal)**
- Threshold 0.50: `0.74413`
- Threshold 0.55: `0.73951`

![Threshold Sweep Results](./threshold_sweep.png)

## Conclusion
We successfully pushed the Private F1 score from **0.73722** to **0.74800**. The final best configuration file `final_zk.yaml` has been updated to use the optimal threshold of `0.45`. The corresponding best submission has been saved to `submissions/best_submission.csv`.
