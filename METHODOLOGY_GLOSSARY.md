# Methodology Glossary — Terms & Techniques Used

This document defines every technical term and method referenced in the pipeline,
written for a reader who may not be familiar with the specific ML techniques.

---

## A. Data Preprocessing Terms

### Reverse Scoring
Some survey questions are phrased negatively (e.g., "Cycling is dangerous" instead of
"Cycling is safe"). To ensure all items point in the same direction, we compute
`6 - original_value` for negatively-worded items. This way, a score of 5 always means
"strongly agrees with cycling adoption" regardless of the original wording.

### Construct Aggregation
Instead of using all 32 individual survey responses as separate features, we average
groups of related items into a single score. For example, ATT1, ATT2, and ATT3 all
measure "Attitude toward cycling," so we average them into one `ATT` score. This
reduces noise from multicollinearity (where highly correlated features confuse the model).

### StandardScaler (Z-Score Normalisation)
Transforms each feature to have mean = 0 and standard deviation = 1. Formula:
`z = (x - mean) / std_dev`. This ensures no feature dominates simply because it has
a larger numeric range.

### Stratified Train/Test Split
Divides the dataset into training (80%) and testing (20%) sets while preserving the
original class proportions in both. Without stratification, the tiny Stage 3 class
(25 samples) might end up with 0 samples in one of the sets.

---

## B. Class Balancing Techniques

### SMOTE (Synthetic Minority Over-sampling Technique)
Creates new synthetic samples for underrepresented classes by finding the k nearest
neighbours of each minority sample and generating new points along the line connecting
them. Think of it as "filling in the gaps" between existing minority examples.

- **Default SMOTE:** Uses k=5 neighbours.
- **Custom SMOTE:** We used k=3 because Stage 3 has only 25 samples — with very few
  samples, fewer neighbours produce more realistic interpolations.

### RandomOverSampler
The simplest balancing method — it randomly duplicates existing minority samples until
the class counts are equal. No new data is created; existing samples are just repeated.
This can lead to overfitting because the model memorises repeated examples.

### Class Weight Balancing
Instead of creating synthetic data, this approach modifies the loss function to penalise
errors on minority classes more heavily. If Stage 3 has 25 samples and Stage 1 has 334,
then misclassifying a Stage 3 sample costs roughly 13× more than misclassifying a
Stage 1 sample. The model is forced to pay more attention to rare classes.

### CTGAN (Conditional Tabular GAN)
A neural network architecture from the NeurIPS 2019 paper. Two networks compete:
- **Generator:** Creates fake tabular rows that look like real data.
- **Discriminator:** Tries to distinguish fake rows from real ones.

Through this adversarial process, the generator learns the true statistical distribution
of the data — including non-linear relationships between features — and can produce
entirely new, realistic samples. This is more sophisticated than SMOTE's linear
interpolation.

### Data Leakage (Why It Matters)
Data leakage occurs when information from outside the training set accidentally influences
model evaluation. In our context: if you apply SMOTE to the entire training set **before**
cross-validation, synthetic samples may be created from data points that end up in both
the training fold and the validation fold. The model effectively gets "hints" about the
validation data, producing inflated scores that do not reflect real-world performance.

**Our solution:** All balancing is enclosed inside `imblearn.Pipeline`, which ensures
SMOTE/CTGAN only sees the training portion of each CV fold.

---

## C. Model Training Terms

### K-Fold Stratified Cross-Validation
The training data is split into k equal-sized "folds" (we use k=5). The model is trained
on 4 folds and validated on the 5th, rotating through all 5 combinations. The final
score is the average across all 5 runs. "Stratified" means each fold preserves the
original class proportions.

### Random Forest
An ensemble of hundreds of decision trees, each trained on a random subset of the data
and features. Final predictions are made by majority vote. The randomisation reduces
overfitting because individual trees' errors cancel out.

### Extra Trees (Extremely Randomised Trees)
Similar to Random Forest, but with an additional layer of randomisation: instead of
finding the *best* split threshold at each node, it picks a random threshold. This
further reduces variance and often performs better on noisy data.

### XGBoost (Extreme Gradient Boosting)
Builds trees sequentially — each new tree focuses on correcting the errors of the
previous trees. Uses gradient descent to minimise a loss function. Powerful but more
prone to overfitting than Random Forest.

### Recursive Feature Elimination (RFE)
An iterative feature selection algorithm. It trains a model, ranks features by
importance, removes the least important feature, and repeats. After enough rounds,
only the most predictive features remain. We use RFE to prune noisy features and
retain only the 5–12 most informative ones.

---

## D. Hyperparameter Tuning Terms

### Bayesian Optimisation (Optuna)
Instead of testing every possible hyperparameter combination (grid search) or testing
random combinations (random search), Bayesian optimisation builds a probabilistic model
of which hyperparameter regions are most promising. It "learns" from previous trials to
focus on the most promising areas of the search space. This typically finds better
solutions in fewer iterations.

### Tree-structured Parzen Estimator (TPE)
The specific algorithm Optuna uses for Bayesian optimisation. It models the probability
distributions of "good" and "bad" hyperparameters separately and samples from regions
where the ratio of good-to-bad is highest.

### Hyperparameters Tuned

| Parameter | What It Controls |
|-----------|-----------------|
| `n_estimators` | Number of trees in the ensemble — more trees = more stable but slower |
| `max_depth` | Maximum depth of each tree — deeper trees capture more complex patterns but risk memorising noise |
| `min_samples_split` | Minimum number of samples required to split a node — higher values prevent splits on tiny, noisy subgroups |
| `min_samples_leaf` | Minimum samples in a leaf node — forces predictions to be based on multiple samples |
| `criterion` | Quality metric for evaluating splits — 'gini' (impurity) or 'entropy' (information gain) |
| `n_features` (RFE) | How many features to keep after elimination |

---

## E. Evaluation Metrics

### F1-Score
The harmonic mean of Precision and Recall: `F1 = 2 × (P × R) / (P + R)`.
A balanced metric that penalises models that sacrifice precision for recall or vice versa.

- **F1-Weighted:** Each class's F1 is weighted by its support (number of test samples).
  Stage 1 (67 samples) contributes more than Stage 3 (5 samples).
- **F1-Macro:** Simply averages all classes' F1 scores equally, regardless of support.
  This harshly penalises poor performance on minority classes.

### Recall (Sensitivity)
The proportion of actual positive cases the model correctly identifies.
`Recall = True Positives / (True Positives + False Negatives)`

### Precision
The proportion of predicted positive cases that are actually positive.
`Precision = True Positives / (True Positives + False Positives)`

### ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
Measures the model's ability to rank predictions correctly across all possible
classification thresholds. A score of 0.90 means that 90% of the time, a randomly
chosen positive sample is ranked higher than a randomly chosen negative sample.

### Log-Loss
Measures how well the model's predicted probabilities match the true labels. Unlike
F1/Recall/Precision (which only care about the final hard prediction), log-loss rewards
confident, correct probabilities and penalises confident, wrong ones.

### Confusion Matrix
A table showing the count of predictions for each true-class/predicted-class combination.
The diagonal contains correct predictions; off-diagonal elements show where the model
confuses one class for another.

---

## F. Explainability Terms

### SHAP (SHapley Additive exPlanations)
A game-theoretic method that assigns each feature a contribution score for each
individual prediction. Based on Shapley values from cooperative game theory — the
"payout" each player (feature) deserves for their contribution to the outcome (prediction).

### SHAP Summary Plot
Shows the overall importance of each feature across all predictions. Features at the
top contribute most to the model's decisions.

### SHAP Dependency Plot
Shows how one specific feature's value (x-axis) affects its SHAP value (y-axis) for
a particular class. This reveals non-linear relationships — for example, "Attitude
scores above 4.0 strongly push toward Stage 5, but below 3.0 have no effect."

### Feature Importance (Gini Importance)
Measures how much each feature reduces impurity (disorder) across all splits in all
trees. Features with higher importance are used more frequently and produce bigger
reductions in classification error. This is a fast, built-in measure but does not
capture feature interactions the way SHAP does.

---

## G. Regularisation Techniques

Regularisation prevents the model from fitting too closely to training data (overfitting).

| Technique | Mechanism |
|-----------|-----------|
| **Max depth** | Limits how deep each tree can grow, forcing broader, more general rules |
| **Min samples leaf/split** | Requires a minimum number of samples before making a decision, preventing decisions based on 1–2 noisy points |
| **Feature pruning (RFE)** | Removes irrelevant features so the model cannot learn from noise |
| **Ensemble averaging** | Averaging hundreds of trees smooths out individual tree errors |

---

*Document generated: April 2026*
