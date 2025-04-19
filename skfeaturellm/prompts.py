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

For each feature provide:
1. A descriptive name that reflects the feature's purpose
2. A clear explanation of what the feature represents and why it's useful
3. A precise formula or logic to create the feature (using Pandas syntax)

Your response should be a always a dictionary with a key called 'ideas' that contains a list of features in JSON format, where each feature has:
- name: A clear, descriptive name
- description: A detailed explanation of the feature
- formula: The exact formula or transformation logic using column names from the dataset. Should have a lambda function that encapsulates the transformation logic. Example: "lambda df: df['column_name'].div(df['other_column_name'])"

Example:
{{
    "ideas": [
        {{
            "name": "credit_score_category",
            "description": "A categorical feature representing the credit score category",
            "formula": "lambda df: pd.cut(df['credit_score'], bins=[0, 600, 700, 800, 900, 1000], labels=['Low', 'Fair', 'Good', 'Very Good', 'Excellent'])"
        }}
    ]
}}
"""
