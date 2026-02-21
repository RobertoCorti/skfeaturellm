import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from skfeaturellm.feature_evaluation.visualizations import plot_feature_vs_target
from skfeaturellm.types import ProblemType

_TEMPLATE_PATH = Path(__file__).parent / "_report_template.html"


class FeatureEvaluationResult:
    """
    Class for storing and presenting feature evaluation results.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        metrics_df: pd.DataFrame,
        primary_metric: Optional[str] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        problem_type: Optional[ProblemType] = None,
    ):
        self._metrics_df = metrics_df
        self._primary_metric = primary_metric
        self._X = X
        self._y = y
        self._problem_type = problem_type

    @property
    def summary(self) -> pd.DataFrame:
        """
        Returns the metrics DataFrame sorted by the primary metric descending.
        """
        sort_col = self._primary_metric or self._metrics_df.columns[0]
        return self._metrics_df.sort_values(by=sort_col, ascending=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary.
        """
        return self._metrics_df.to_dict()

    def to_html(self, path: str) -> None:
        """
        Save a self-contained HTML report to disk.

        Parameters
        ----------
        path : str
            File path where the HTML report will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._create_report())

    def _create_report(self) -> str:
        """Build and return the full HTML report as a string."""
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        return template.replace("{{FEATURE_CARDS}}", self._build_feature_cards())

    def _build_feature_cards(self) -> str:
        """Build one card per feature, ordered by primary metric descending."""
        cards = ""
        for feature in self.summary.index:
            plot_html = self._build_plot_html(feature)
            stats_html = self._build_stats_html(feature)
            cards += (
                f"<section class='feature-card'>"
                f"<h3 class='card-header'>{feature}</h3>"
                f"<div class='card-body'>{plot_html}{stats_html}</div>"
                f"</section>\n"
            )
        return cards

    def _build_plot_html(self, feature: str) -> str:
        """Generate a base64-embedded plot for a single feature, or empty string."""
        if self._X is None or self._y is None or self._problem_type is None:
            return ""
        fig = plot_feature_vs_target(self._X[feature], self._y, self._problem_type)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return (
            f"<div class='card-plot'>"
            f'<img src="data:image/png;base64,{img_b64}" alt="{feature}">'
            f"</div>"
        )

    def _build_stats_html(self, feature: str) -> str:
        """Generate a vertical key-value stats list for a single feature."""
        rows = ""
        for metric, value in self._metrics_df.loc[feature].items():
            if isinstance(value, float):
                formatted = f"{value:.4f}" if not pd.isna(value) else "N/A"
            else:
                formatted = str(value)
            rows += (
                f"<div class='stat-row'>"
                f"<span class='stat-label'>{metric}</span>"
                f"<span class='stat-value'>{formatted}</span>"
                f"</div>"
            )
        return f"<div class='card-stats'>{rows}</div>"

    def __repr__(self) -> str:
        return self.summary.__repr__()
