"""Cohort characteristics testing (manuscript Table 1)."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kstest, levene, ttest_ind
from statsmodels.stats.contingency_tables import Table2x2

from sga.config import LABEL


def normality_and_parametric_test(
    df, df_sga, df_non_sga, feature, continuous=True, results=None
):
    """Test one variable for a group difference and append a Table 1 row."""
    results = [] if results is None else results
    df = df.dropna(subset=[feature]).reset_index(drop=True)
    df_sga = df_sga[feature].dropna().reset_index(drop=True)
    df_non_sga = df_non_sga[feature].dropna().reset_index(drop=True)

    # Test against a normal fitted to each group, not against the standard normal:
    _, sga_p = kstest(df_sga, "norm", args=(df_sga.mean(), df_sga.std(ddof=1)))
    _, not_sga_p = kstest(
        df_non_sga, "norm", args=(df_non_sga.mean(), df_non_sga.std(ddof=1))
    )
    print("SGA Kolmogorov-Smirnov Test: ", sga_p)
    print("Not SGA Kolmogorov-Smirnov Test: ", not_sga_p)

    if continuous:
        # Calculate means and percentage difference
        mean_sga = df_sga.mean()
        mean_non_sga = df_non_sga.mean()
        percentage_diff = (abs(mean_sga - mean_non_sga) / mean_non_sga) * 100

        print(f"Mean of SGA group: {mean_sga:.3f}")
        print(f"Mean of Non-SGA group: {mean_non_sga:.3f}")
        print(f"Percentage difference: {percentage_diff:.2f}%")

        # Table 1 and the Methods both state Student's t-test, so that is the
        # reported p-value. Welch's is computed alongside and reported as a
        # sensitivity check, together with Levene's test of the equal-variance
        # assumption Student's t-test rests on: where the two p-values straddle
        # 0.05 the choice of test matters and should be stated explicitly.
        _, levene_p = levene(df_sga, df_non_sga)
        equal_var = bool(levene_p > 0.05)

        _, p_value = ttest_ind(df_sga, df_non_sga, equal_var=True)
        _, welch_p_value = ttest_ind(df_sga, df_non_sga, equal_var=False)
        print("Equal Variance (Levene p > 0.05): ", equal_var, f"(p={levene_p:.4g})")
        print("Student's t-test p-value: ", p_value)
        print("Welch's t-test p-value (sensitivity check): ", welch_p_value)
        if (p_value < 0.05) != (welch_p_value < 0.05):
            print(
                "  [warn] Student's and Welch's t-tests disagree at alpha=0.05 for "
                f"{feature}; report which test was used."
            )
        test_used = "Student t-test"

        # Calculate 95% confidence interval (unequal variance)
        n_sga = len(df_sga)
        n_non_sga = len(df_non_sga)
        var_sga = np.var(df_sga, ddof=1)
        var_non_sga = np.var(df_non_sga, ddof=1)
        se_diff = np.sqrt(
            var_sga / n_sga + var_non_sga / n_non_sga
        )  # Standard error for unequal variances
        ci_lower = (mean_sga - mean_non_sga) - 1.96 * se_diff
        ci_upper = (mean_sga - mean_non_sga) + 1.96 * se_diff

        print(
            f"95% Confidence Interval (Unequal Variance): [{ci_lower:.3f}, {ci_upper:.3f}]"
        )

        # Compute Cohen's d for practical significance
        mean_diff = abs(df_sga.mean() - df_non_sga.mean())
        pooled_std = np.sqrt(
            (
                (len(df_sga) - 1) * np.var(df_sga, ddof=1)
                + (len(df_non_sga) - 1) * np.var(df_non_sga, ddof=1)
            )
            / (len(df_sga) + len(df_non_sga) - 2)
        )
        cohen_d = mean_diff / pooled_std
        print(f"Effect Size (Cohen's d): {cohen_d:.3f}")

        # Interpret the effect size
        if cohen_d < 0.2:
            significance = "negligible"
        elif cohen_d < 0.5:
            significance = "small"
        elif cohen_d < 0.8:
            significance = "medium"
        else:
            significance = "large"
        print(f"Practical Significance: The effect size is {significance}.")

        results.append({
            "Feature": feature,
            "Test Used": test_used,
            "P-value": p_value,
            "Welch P-value": welch_p_value,
            "Levene P-value": levene_p,
            "Equal Variance": equal_var,
            "Percentage Difference": percentage_diff,
            "Confidence Interval": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            "Effect Size (Cohen's d)": cohen_d,
            "Effect Significance": significance
        })
    else:
        # Categorical variable processing
        contingency_table = pd.crosstab(df[LABEL], df[feature])
        print("\nContingency Table:")
        print(contingency_table)

        # Chi-square test
        chi2_stat, p, dof, expected = chi2_contingency(contingency_table)
        print("Chi-square Test p-value:", p)

        results.append({
            "Feature": feature,
            "Test Used": "Chi-square Test",
            "P-value": p
        })

        # Odds ratio for each class
        for category in contingency_table.columns:
            # Create 2x2 table for the current category
            in_class = contingency_table[category].to_numpy()
            not_in_class = contingency_table.sum(axis=1) - in_class
            table_2x2 = np.array(
                [in_class, not_in_class]
            ).T  # Rows: SGA, Non-SGA | Cols: In-Class, Not-In-Class

            # Compute odds ratio
            if table_2x2.shape == (2, 2):  # Ensure it's a valid 2x2 table
                odds_ratio_result = Table2x2(table_2x2)
                print(f"Category: {category}")
                print(f"Odds Ratio (OR): {odds_ratio_result.oddsratio:.3f}")
                print(
                    f"95% CI: [{odds_ratio_result.oddsratio_confint()[0]:.3f}, {odds_ratio_result.oddsratio_confint()[1]:.3f}]"
                )
                results.append({
                    "Feature": feature,
                    "Category": category,
                    "Odds Ratio": odds_ratio_result.oddsratio,
                    "95% CI": f"[{odds_ratio_result.oddsratio_confint()[0]:.3f}, {odds_ratio_result.oddsratio_confint()[1]:.3f}]"
                })
    return results


def get_count_categorical_feature(positive_df, negative_df, feature, results=None):
    """Append per-category counts and percentages for one categorical variable."""
    results = [] if results is None else results
    positive_count = positive_df[feature].value_counts()
    negative_count = negative_df[feature].value_counts()
    
    # Compute percentage within each group
    positive_percentage = (positive_count / positive_count.sum() * 100).fillna(0)
    negative_percentage = (negative_count / negative_count.sum() * 100).fillna(0)

    print(f"Positive {feature} Counts and Percentage:")
    print(pd.DataFrame({'Count': positive_count, 'Percentage': positive_percentage}).to_string())

    print(f"\nNegative {feature} Counts and Percentage:")
    print(pd.DataFrame({'Count': negative_count, 'Percentage': negative_percentage}).to_string())

    for category in positive_count.index:
        results.append({
            "Feature": feature,
            "Category": category,
            "Group": "Positive",
            "Count": positive_count[category],
            "Percentage": positive_percentage[category]
        })

    for category in negative_count.index:
        results.append({
            "Feature": feature,
            "Category": category,
            "Group": "Negative",
            "Count": negative_count[category],
            "Percentage": negative_percentage[category]
        })
    return results


def get_mean_continuous_feature(positive_df, negative_df, feature, results=None):
    """Append mean, SD and per-group count for one continuous variable."""
    results = [] if results is None else results
    positive_count = positive_df[feature].count()
    negative_count = negative_df[feature].count()

    positive_mean = positive_df[feature].mean()
    positive_std = positive_df[feature].std()

    negative_mean = negative_df[feature].mean()
    negative_std = negative_df[feature].std()

    print(
        f"\nPositive SGA {feature}: Count - {positive_count}, Mean =",
        positive_mean,
        ", Std =",
        positive_std,
    )
    print(
        f"Negative SGA {feature}: Count - {negative_count}, Mean =",
        negative_mean,
        ", Std =",
        negative_std,
    )

    results.append({
        "Feature": feature,
        "Group": "Positive",
        "Count": positive_count,
        "Mean": positive_mean,
        "Std": positive_std
    })
    results.append({
        "Feature": feature,
        "Group": "Negative",
        "Count": negative_count,
        "Mean": negative_mean,
        "Std": negative_std
    })
    return results


def save_results_to_csv(data, filename="analysis_results.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def save_results_to_excel(data, filename="analysis_results.xlsx"):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
