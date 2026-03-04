"""
Demo: preview the full LLM prompt with dataset statistics injected.

Run with:
    poetry run python demo_statistics.py
"""

from unittest.mock import patch

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.types import ProblemType

SEP = "=" * 60


def show_full_prompt(title, X, y, problem_type):
    print(SEP)
    print(title)
    print(SEP)

    with patch("skfeaturellm.llm_interface.init_chat_model"):
        llm = LLMInterface()

    feature_descriptions = [
        {"name": col, "type": str(X[col].dtype), "description": ""} for col in X.columns
    ]
    dataset_statistics = LLMInterface._format_dataset_statistics(X, y, problem_type)
    context = llm.generate_prompt_context(
        feature_descriptions=feature_descriptions,
        problem_type=problem_type.value,
        dataset_statistics=dataset_statistics,
    )

    messages = llm.prompt_template.format_messages(**context)
    for msg in messages:
        print(f"[{msg.__class__.__name__}]\n{msg.content}")
    print()


# --- Regression: diabetes dataset ---
diabetes = load_diabetes()
X_reg = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_reg = pd.Series(diabetes.target, name="progression")
show_full_prompt(
    "REGRESSION — sklearn diabetes dataset", X_reg, y_reg, ProblemType.REGRESSION
)

# --- Classification: breast cancer dataset ---
cancer = load_breast_cancer()
X_clf = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_clf = pd.Series(cancer.target_names[cancer.target], name="diagnosis")
show_full_prompt(
    "CLASSIFICATION — sklearn breast cancer dataset",
    X_clf,
    y_clf,
    ProblemType.CLASSIFICATION,
)
