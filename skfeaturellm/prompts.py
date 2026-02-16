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

For UNARY operations (log, sqrt, abs, etc.):
- column: The name of the column to transform

For BINARY operations (add, sub, mul, div):
- left_column: The name of the first column (left operand)
- right_column: The name of the second column (for column-column operations) - OR -
- right_constant: A numeric constant (for column-constant operations)

NOTE: For binary operations, provide EITHER right_column OR right_constant, never both.

## Examples

Unary operation (log transformation):
{{
    "type": "log",
    "feature_name": "log_income",
    "description": "Natural log of income to reduce right skewness and stabilize variance",
    "column": "annual_income"
}}

Binary column-column operation (ratio):
{{
    "type": "div",
    "feature_name": "debt_to_income_ratio",
    "description": "Ratio of total debt to annual income, indicating financial leverage",
    "left_column": "total_debt",
    "right_column": "annual_income"
}}

Binary column-constant operation (scaling):
{{
    "type": "mul",
    "feature_name": "monthly_income",
    "description": "Annual income converted to monthly by dividing by 12",
    "left_column": "annual_income",
    "right_constant": 0.0833
}}

Make sure to use the EXACT column names from the dataset provided above.
"""
