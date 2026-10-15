"""Eligibility screening: CONSORT-style accounting and IQR outlier filtering."""

from __future__ import annotations

import pandas as pd


class ScreeningLog:
    """Ordered record of how many records each eligibility step removed."""

    def __init__(self, cohort: str, screened: int):
        self.cohort = cohort
        self.screened = int(screened)
        self.remaining = int(screened)
        self.steps: list[tuple[str, int, int]] = []

    def record(self, reason: str, remaining: int) -> int:
        """Log one exclusion step given the number of records left after it."""
        remaining = int(remaining)
        self.steps.append((reason, self.remaining - remaining, remaining))
        self.remaining = remaining
        return remaining

    def apply(self, reason: str, df: pd.DataFrame) -> pd.DataFrame:
        """Log a step from the size of the frame it produced."""
        self.record(reason, len(df))
        return df

    def to_frame(self) -> pd.DataFrame:
        """Return the flow as a tabular ``reason / excluded / remaining`` frame."""
        return pd.DataFrame(
            self.steps, columns=["exclusion_reason", "n_excluded", "n_remaining"]
        )

    def report(self, entering: int | None = None, extra: dict | None = None) -> None:
        """Print the CONSORT-style flow for this cohort."""
        entering = self.remaining if entering is None else int(entering)
        width = 78
        print("\n" + "=" * width)
        print(f"CONSORT-STYLE COHORT FLOW - {self.cohort}")
        print("=" * width)
        print(f"  {'Records screened':<56}{self.screened:>10}")
        print("  " + "-" * (width - 4))
        for reason, excluded, remaining in self.steps:
            print(f"  - excluded: {reason:<44}{excluded:>10}  ({remaining} left)")
        print("  " + "-" * (width - 4))
        print(f"  {'Entering training / testing':<56}{entering:>10}")
        for key, value in (extra or {}).items():
            print(f"  {key:<56}{value:>10}")
        print("=" * width)


def filter_df_IQR(df, label="sga"):
    """Split a frame into interquartile-range inliers and outliers."""
    labels = list(label) if isinstance(label, (list, tuple)) else [label]
    columns = [c for c in df.columns if c not in labels and c not in ("id", "gender")]

    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    within = df[columns].isna() | ((df[columns] >= lower) & (df[columns] <= upper))
    keep = within.all(axis=1)

    inliers, outliers = df[keep], df[~keep]
    print(
        f"IQR screening: {len(df)} records -> {len(inliers)} inliers, "
        f"{len(outliers)} outliers"
    )
    for name, part in (("inlier", inliers), ("outlier", outliers)):
        for column in labels:
            if column in part.columns and part[column].nunique() == 1:
                raise ValueError(
                    f"The {name} partition has only one distinct {column!r} value."
                )
    return inliers, outliers
