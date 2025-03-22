"""
Module containing prompts for LLM interactions.
"""

FEATURE_ENGINEERING_PROMPT = """You are an expert data scientist specializing in feature engineering for tabular data.
Given the following dataset information, suggest meaningful features that could improve model performance.

Dataset Information:
{data_description}

{target_description}

Additional Context:
{additional_context}

Generate feature engineering ideas that:
1. Are relevant to the problem
2. Use appropriate transformations based on the data types
3. Capture meaningful patterns and relationships
4. Are computationally feasible

For each feature, provide:
1. A descriptive name
2. A clear explanation of what it represents
3. The type of feature (numerical, categorical, etc.)
4. The input columns required

Respond in a structured format that can be parsed into the following schema:
{feature_schema}
""" 