import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper


def regress_and_analyze_features(
    sequence_weights: np.ndarray, patristic_distances: np.ndarray,
) -> tuple[RegressionResultsWrapper, pd.DataFrame]:
    """Performs linear regression and a Type III ANOVA for feature importance.

    This function fits a linear model then returns returns the statsmodels OLS model and
    a DataFrame with Type III ANOVA results, which assesses the unique contribution of
    each feature.

    Args:
        sequence_weights: The feature matrix (N, 22).
        patristic_distances: The target vector (N,).

    Returns:
        tuple: A tuple containing:
            - model: The fitted statsmodels OLS results object.
            - anova_results: DataFrame with Type III ANOVA results.
    """
    assert sequence_weights.shape[1] == 22
    assert sequence_weights.shape[0] == patristic_distances.shape[0]

    X = sequence_weights
    y = patristic_distances
    _, k = X.shape

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_with_const = sm.add_constant(X_normalized, prepend=True)
    sm_model = sm.OLS(y_normalized, X_with_const).fit()

    anova_results_list = []
    for i in range(k):
        mask = np.ones(k+1, dtype=bool)  # Add 1 to account for constant term
        mask[i+1] = False

        X_reduced = X_with_const[:, mask]
        sm_model_reduced = sm.OLS(y_normalized, X_reduced).fit()

        ss_reduced = sm_model_reduced.ssr
        ss_full = sm_model.ssr

        ss_feature = ss_reduced - ss_full

        ms_feature = ss_feature / 1
        ms_error = sm_model.mse_resid
        f_stat = ms_feature / ms_error

        p_value = 1 - stats.f.cdf(f_stat, 1, sm_model.df_resid)

        anova_results_list.append(
            {
                "sum_of_squares": ss_feature,
                "F": f_stat,
                "p_value": p_value,
            }
        )

    anova_results = pd.DataFrame(anova_results_list, index=pd.Series(list(range(k))))

    total_explained_ss = anova_results["sum_of_squares"].sum()
    frac_explained_ss = anova_results["sum_of_squares"] / total_explained_ss
    anova_results["percent_sum_sq"] = frac_explained_ss * 100
    anova_results["adj_r2_contrib"] = frac_explained_ss * sm_model.rsquared_adj

    return sm_model, anova_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_samples = 150
    num_features = 22
    np.random.seed(42)

    X_data = np.random.rand(num_samples, num_features)
    y_data = (
        3.5 * X_data[:, 2]
        - 2.1 * X_data[:, 10]
        + 0.8 * X_data[:, 15]
        + np.random.randn(num_samples) * 2
    )

    model, anova_importance = regress_and_analyze_features(X_data, y_data)

    print(f"R²: {model.rsquared:.4f}")
    print(f"Adjusted R²: {model.rsquared_adj:.4f}\n")

    print("--- Type III ANOVA Feature Importance ---")
    print(anova_importance.sort_values(by="F", ascending=False).round(4))

    y_pred = model.fittedvalues
    y_actual = model.model.endog

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_actual, y_pred, alpha=0.5, s=50)

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Predicted vs Actual\nR² = {model.rsquared:.4f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
