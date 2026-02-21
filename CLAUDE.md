# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SKFeatureLLM is a Python library for LLM-powered feature engineering with a scikit-learn compatible API. It uses LLMs (via LangChain) to generate feature engineering ideas, applies them as transformations, and evaluates the resulting features.

- **Version:** 0.1.0
- **Python:** ≥ 3.10
- **License:** MIT
- **Package manager:** Poetry
- **Authors:** Roberto Corti, Stefano Polo

## Commands

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_feature_engineer.py

# Run a specific test
poetry run pytest tests/test_feature_engineer.py::test_initialization -v

# Formatting
poetry run black .
poetry run isort .

# Linting and type checking
poetry run pylint skfeaturellm
poetry run mypy skfeaturellm

# Pre-commit hooks (runs black, isort, trailing whitespace, end-of-file fixes)
poetry run pre-commit run --all-files

# Build docs
cd docs && make html
```

## Architecture

The system has four main layers:

1. **Orchestrator** — `LLMFeatureEngineer` (`feature_engineer.py`) extends scikit-learn's `BaseEstimator`/`TransformerMixin`. Entry point that ties all layers together via `fit()`, `transform()`, and `evaluate_features()`.

2. **LLM Layer** — `LLMInterface` (`llm_interface.py`) communicates with LLMs using LangChain's `init_chat_model()` and `with_structured_output()` to get validated responses matching Pydantic schemas.

3. **Transformation Layer** — `transformations/` uses a registry pattern (`@register_transformation` decorator on `BaseTransformation` subclasses). `TransformationExecutor` parses configs and applies registered transformations.

4. **Evaluation Layer** — `feature_evaluation/` contains `FeatureEvaluator` (delegates to pure-function metrics in `metrics.py`), and `FeatureEvaluationResult` (wraps results in a DataFrame with summary/sorting).

**Key data flow:** `LLMFeatureEngineer.fit()` → LLM generates `FeatureEngineeringIdeas` (validated by Pydantic schemas) → `transform()` converts each idea via `to_executor_dict()` into `TransformationExecutor` → executor produces new DataFrame columns → `FeatureEvaluator` scores them.

## Project Structure

```
skfeaturellm/
├── __init__.py                  # Exports LLMFeatureEngineer
├── feature_engineer.py          # LLMFeatureEngineer (orchestrator)
├── llm_interface.py             # LLMInterface (LangChain wrapper)
├── schemas.py                   # Pydantic v2 models for LLM I/O
├── prompts.py                   # LLM prompt templates
├── types.py                     # ProblemType enum
├── reporting.py                 # FeatureReport (not yet implemented)
├── feature_evaluation/
│   ├── __init__.py              # Exports FeatureEvaluator, FeatureEvaluationResult
│   ├── evaluator.py             # FeatureEvaluator (orchestrates metrics)
│   ├── metrics.py               # Pure-function metric computations
│   └── result.py                # FeatureEvaluationResult container
└── transformations/
    ├── __init__.py              # Re-exports + triggers registration
    ├── base.py                  # BaseTransformation ABC, error classes
    ├── executor.py              # TransformationExecutor, registry, parsers
    ├── unary/
    │   ├── __init__.py
    │   └── arithmetic.py        # log, log1p, abs, exp, pow
    └── binary/
        ├── __init__.py
        ├── arithmetic.py        # add, sub, mul, div
        └── errors.py            # Binary-specific errors
```

## Key Patterns & Conventions

### Transformation Registry

New transformations are registered at import time via a decorator in `executor.py`:

```python
@register_transformation("log")
class LogTransformation(UnaryTransformation):
    ...
```

The registry (`_TRANSFORMATION_REGISTRY`) is populated when `transformations/__init__.py` imports the submodules. The executor looks up classes by name string from this dict. When adding a new transformation, create the class with `@register_transformation`, then import it in the appropriate `__init__.py`.

**Current operations:** unary (`log`, `log1p`, `abs`, `exp`, `pow`) and binary (`add`, `sub`, `mul`, `div`).

### Schemas & Validation

`schemas.py` defines Pydantic v2 models with `extra="forbid"` on `TransformationParameters`. `FeatureEngineeringIdea` uses a `@model_validator(mode="after")` to ensure transformation types are registered and operand counts match (unary=1 column, binary=1 or 2 columns). `FeatureEngineeringIdeas` is the top-level structured output schema the LLM must conform to.

### Error Handling

- Custom exception hierarchy rooted at `TransformationError` (in `transformations/base.py`): `ColumnNotFoundError`, `TransformationParseError`, `InvalidValueError`, `DivisionByZeroError`.
- Uses `warnings.warn()` throughout (not `logging`). This includes the executor when skipping failed transformations (`raise_on_error=False`) and `metrics.mutual_information()` when computation fails for a column.
- Graceful degradation: `TransformationExecutor` can skip failures, and metric functions return `np.nan` on error.

### Problem Types

`ProblemType` enum in `types.py` (`CLASSIFICATION` / `REGRESSION`) drives the LLM prompt context, which evaluation metrics are computed (regression adds Spearman/Pearson correlation), and which mutual information function (`mutual_info_classif` vs `mutual_info_regression`) is used.

### LLM Integration

`LLMInterface` wraps LangChain's `init_chat_model()` → `with_structured_output(FeatureEngineeringIdeas)` → chains with `ChatPromptTemplate` using the `|` operator. Any model/provider supported by LangChain's `init_chat_model()` works (default: `gpt-4o`). Extra kwargs (e.g., `temperature`, `model_provider`) are passed through.

## Testing

```
tests/
├── __init__.py
├── test_feature_engineer.py       # Orchestrator tests
├── test_llm_interface.py          # LLM layer tests
├── feature_evaluation/
│   ├── conftest.py                # Fixtures: sample_data_regression, sample_data_classification
│   └── test_evaluator.py          # Evaluator + result tests
└── transformations/
    ├── __init__.py
    ├── conftest.py                # Fixtures: sample_df, sample_config
    ├── test_executor.py           # Executor + registry tests
    ├── test_unary.py              # Unary transformation tests
    └── test_binary.py             # Binary transformation tests
```

**Conventions:**
- Function-based tests (no test classes), named `test_*`.
- Shared fixtures live in `conftest.py` files within each test subpackage.
- LLM calls are always mocked — mock `init_chat_model` to avoid real API calls, then mock `chain.invoke` return values with `FeatureEngineeringIdea` Pydantic objects.
- Uses `pytest-mock` (`mocker` fixture) and `unittest.mock.Mock`.
- Uses `np.testing.assert_array_almost_equal()` for float comparisons.
- Error cases tested with `pytest.raises()`, warnings with `pytest.warns()`.
- pytest config in `pytest.ini`: `testpaths = tests`, `addopts = -v --tb=short`.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) with four jobs:

1. **pre-commit** — Runs `pre-commit run --all-files` (black, isort, trailing whitespace, end-of-file fixer).
2. **run-tests** — Runs `poetry run pytest`.
3. **create-tag** — On push to `main`, auto-increments patch version and creates a git tag.
4. **publish** — Builds and publishes to PyPI via trusted publishing.

## Style

- **Black:** line-length 88, preview mode, target py310/py311
- **isort:** black profile, trailing comma, group by package
- **mypy:** strict mode, show error context/codes/column numbers, warn unreachable
- **pylint:** max line-length 100, fail-under 10, snake_case functions/variables, PascalCase classes, UPPER_CASE constants, max-args=5, max-attributes=7
- **Pre-commit hooks:** black, isort, trailing-whitespace, end-of-file-fixer (pylint and mypy hooks are commented out)

## Dependencies

**Runtime:** scikit-learn, pandas, numpy, pydantic (v2), langchain, langchain-openai, langchain-classic

**Dev:** pytest, pytest-mock, pytest-cov, mypy, pylint, black, isort, pre-commit, sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints, nbsphinx, jupyter

**Docs:** Sphinx with Read the Docs theme, hosted via `.readthedocs.yaml`. Docs source in `docs/`, built with `make html`.
