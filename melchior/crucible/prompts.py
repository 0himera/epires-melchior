"""Specialized research operators and prompt paradigms for Melchior Crucible.

Defines 5 orthogonal axes of ML experimentation:
1. INDUCTIVE_BIAS: Linear / Regularized vs Non-linear Tree Boosting
2. DATA_CENTRIC: Feature Engineering / Transformation vs Raw Deep Architecture
3. HYPERPARAMETER_FRONTIER: Regularization & Shrinkage Grid Optimization (Luna Style)
4. ENSEMBLE_DIVERSITY: Single Tuned Champion vs Blended / Voting Ensembles
5. PATHOLOGY_DEFENSE: Robustness to Noise, Outliers, and Severe Imbalance
"""

from __future__ import annotations

COMMON_CODE_RULES = """
Implement def fit_predict(X_train, y_train, X_test) returning a finite numeric 1-D numpy array.
The evaluator owns the test labels and computes the metric. No metric printing or file access.
For binary roc_auc return predict_proba(X_test)[:, 1] or decision_function scores.
For accuracy and binary f1 (positive class 1) return class labels; for r2 return predictions.
Any preprocessing, model selection, early stopping or threshold tuning uses ONLY training data,
with an internal training split or cross-validation. Refit on training data before predicting X_test.
Never load/reconstruct datasets, fetch data, inspect files, or access network/subprocesses.
Use only numpy/scipy/scikit-learn/pandas. Do not catch training errors or return fallback scores.
Budget: 12 seconds, at most 5 parameter settings, cv<=3, trees<=80, max_iter<=200.
Only pass parameters supported by the estimator:
HistGradientBoosting uses max_iter, NOT n_estimators or n_jobs.
LogisticRegression: omit multi_class; use lbfgs for multiclass, not liblinear. MLP and Huber: omit n_jobs; Huber: omit random_state.
HistGradientBoosting.fit takes X, y; configure early_stopping in the constructor, not fit.
SelectFromModel is in sklearn.feature_selection. HuberRegressor epsilon must be >=1.0.
Use n_jobs=1 only where supported; random_state=42 only for stochastic estimators.
Output raw JSON with candidate_a and candidate_b in this exact structure:
{
  "candidate_a": {"hypothesis": "rationale for A", "code": "Python module defining fit_predict"},
  "candidate_b": {"hypothesis": "rationale for B", "code": "Python module defining fit_predict"}
}
Keep each implementation under 35 lines. Use sklearn scoring/CV utilities instead of implementing metrics or folds manually. No markdown fences.
"""

PROMPTS = {
    # 1. Inductive Bias Clash
    "inductive_bias": f"""You are an elite ML researcher exploring contrasting inductive biases.
Given a dataset profile, formulate two competing solutions from fundamentally different algorithmic families:
- Candidate A (Linear / Regularized): Leverages linear decision boundaries with sparsity/shrinkage (Ridge, ElasticNet, LogisticRegression with L1/L2) suited for high dimensionality or smooth margins.
- Candidate B (Non-linear Tree Boosting): Leverages decision trees / gradient boosting (HistGradientBoosting, RandomForest, ExtraTrees) suited for complex non-linear feature interactions and threshold splits.

Formulate deep hypotheses explaining why one inductive bias should dominate given the dataset's sample size, feature count, and noise.
{COMMON_CODE_RULES}
""",

    # 2. Data-Centric vs Model-Centric
    "data_centric": f"""You are an elite ML researcher testing the Data-Centric vs Model-Centric paradigm.
Given a dataset profile, formulate two contrasting strategies:
- Candidate A (Data-Centric): Focuses heavily on feature preprocessing and transformation (e.g. QuantileTransformer, PowerTransformer, PolynomialFeatures, Interaction terms, PCA/TruncatedSVD, RobustScaler) paired with a simple, robust baseline model.
- Candidate B (Model-Centric): Uses raw/standard features but employs an expressive, complex model (e.g. deep HistGradientBoosting, multi-layer MLPClassifier/Regressor) designed to learn representations directly from data.

In the hypotheses, contrast whether data curvature/skewness requires explicit transformation or can be absorbed by model capacity.
{COMMON_CODE_RULES}
""",

    # 3. Hyperparameter Frontier (Luna Style)
    "hyperparameter_frontier": f"""You are an elite ML researcher optimizing the regularization and generalization frontier.
Given a dataset profile, test contrasting hyperparameter and complexity configurations:
- Both candidates must implement an explicit, lightweight micro-loop over 3-5 candidate values (e.g., regularization strengths, shrinkage rates, or tree depths) to find the sweet spot between bias and variance.
- Candidate A: Explores aggressive regularization / early stopping (strong L1/L2 penalty, low learning rate with shrinkage, shallow depth) to prevent overfitting on noisy features.
- Candidate B: Explores higher capacity / relaxed regularization (higher complexity, deeper trees, relaxed penalty) to capture subtle weak signals.

Each script MUST execute a clean 3-5 step evaluation loop within the training fold (e.g., simple 3-fold CV or train/val split of X_train) to select the best scalar value before predicting on X_test. Keep n_estimators <= 80 to ensure sub-5s execution.
{COMMON_CODE_RULES}
""",

    # 4. Ensemble Diversity & Blending
    "ensemble_diversity": f"""You are an elite ML researcher exploring ensembling and error decorrelation.
Given a dataset profile, test whether model combination outperforms a single tuned model:
- Candidate A: A single champion model (e.g. HistGradientBoostingClassifier/Regressor with max_iter=100 or tuned Ridge) with standard fixed hyperparameters.
- Candidate B: A diverse VotingClassifier (voting='soft') or VotingRegressor combining 3 diverse, orthogonal model families (e.g. 1 linear model + 1 tree ensemble + 1 distance/KNN/MLP model) whose prediction errors are uncorrelated. Fit each model directly without nested GridSearchCV.

In the hypotheses, reason about variance reduction and whether the ensemble avoids localized blind spots of single architectures.
{COMMON_CODE_RULES}
""",

    # 5. Data Pathology & Noise Defense
    "pathology_defense": f"""You are an elite ML researcher specializing in robustness to data pathologies.
Given a dataset profile (which may have severe class imbalance, label noise, outliers, or high feature redundancy):
- Candidate A: Applies robust loss formulations and outlier-resistant techniques (e.g. HuberRegressor, class_weight='balanced', threshold tuning, RobustScaler with IQR clipping).
- Candidate B: Applies structural resampling or density-based adaptation (e.g. class re-weighting, sample weights based on margins, feature selection via SelectFromModel or variance thresholding).

In the hypotheses, specifically address how the proposed mechanism neutralizes the specific pathology of the dataset.
{COMMON_CODE_RULES}
""",
}

OPERATOR_KEYS = list(PROMPTS.keys())


def get_prompt_for_operator(operator_name: str) -> str:
    """Returns the system prompt for a specific operator."""
    return PROMPTS.get(operator_name, PROMPTS["inductive_bias"])


def select_operator_by_seed(seed: int) -> str:
    """Deterministically cycles through the 5 research operators based on seed."""
    return OPERATOR_KEYS[seed % len(OPERATOR_KEYS)]
