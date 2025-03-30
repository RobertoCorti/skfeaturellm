"""
Module containing prompts for LLM interactions.
"""

# pylint: disable=line-too-long

FEATURE_ENGINEERING_PROMPT = """You are an expert data scientist specializing in feature engineering for tabular data.
Given the following dataset information, suggest meaningful features that could improve model performance.

Dataset Information:
{feature_descriptions}

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
3. A precise formula or logic to create the feature (using Python/Pandas syntax)

Your response should be a list of features in JSON format, where each feature has:
- name: A clear, descriptive name
- description: A detailed explanation of the feature
- formula: The exact formula or transformation logic using column names from the dataset
"""
