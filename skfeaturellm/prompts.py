"""
Module containing prompts for LLM interactions.
"""

# pylint: disable=line-too-long

FEATURE_ENGINEERING_PROMPT = """You are an expert data scientist specializing in feature engineering for tabular data.
Given the following dataset information, suggest meaningful features that could improve model performance.

Dataset Information:
{feature_descriptions}

Problem Type:
{problem_type}

Target Description:
{target_description}

Additional Context:
{additional_context}

Generate feature engineering ideas that:
1. Are relevant to the problem
2. Use appropriate transformations based on the data types
3. Capture meaningful patterns and relationships
4. Are computationally feasible

IMPORTANT: You must use the structured transformation format described below.

## Supported Transformation Types

{transformation_types}

## Required Fields for Each Feature

All transformations require:
- type: The transformation type (see above)
- feature_name: A clear, descriptive name for the new feature
- description: A detailed explanation of what the feature represents and why it's useful
- columns: A list of column names required for the transformation

For UNARY operations (log, sqrt, abs, etc.):
- columns: A list with exactly 1 column name

For BINARY operations (add, sub, mul, div):
- columns: A list with 1 or 2 column names
  - For column-column operations: provide 2 column names
  - For column-constant operations: provide 1 column name + parameters with "constant"

## Examples

Unary operation (log transformation):
{{
    "type": "log",
    "feature_name": "log_income",
    "description": "Natural log of income to reduce right skewness and stabilize variance",
    "columns": ["annual_income"]
}}

Binary column-column operation (ratio):
{{
    "type": "div",
    "feature_name": "debt_to_income_ratio",
    "description": "Ratio of total debt to annual income, indicating financial leverage",
    "columns": ["total_debt", "annual_income"]
}}

Binary column-constant operation (scaling):
{{
    "type": "mul",
    "feature_name": "monthly_income",
    "description": "Annual income converted to monthly by dividing by 12",
    "columns": ["annual_income"],
    "parameters": {{"constant": 0.0833}}
}}

Make sure to use the EXACT column names from the dataset provided above.
"""
