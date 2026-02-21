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
        table_html = self.summary.to_html(float_format="{:.4f}".format)
        plots_section = self._build_plots_section()
        return template.replace("{{TABLE}}", table_html).replace(
            "{{PLOTS}}", plots_section
        )

    def _build_plots_section(self) -> str:
        """Generate base64-embedded plot images if context data is available."""
        if self._X is None or self._y is None or self._problem_type is None:
            return ""

        imgs_html = ""
        for feature in self._X.columns:
            fig = plot_feature_vs_target(self._X[feature], self._y, self._problem_type)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            imgs_html += (
                f"<figure>"
                f'<img src="data:image/png;base64,{img_b64}" alt="{feature}">'
                f"<figcaption>{feature}</figcaption>"
                f"</figure>\n"
            )

        return f"<h2>Feature Distributions</h2>\n<div class='plots'>\n{imgs_html}</div>"

    def __repr__(self) -> str:
        return self.summary.__repr__()
