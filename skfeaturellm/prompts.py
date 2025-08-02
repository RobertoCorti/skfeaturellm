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
- formula: The exact formula or transformation logic using column names from the dataset.
          It has to follow this syntax: "A+B'" This expresion will be taken by Pandas' DataFrame.eval() method.
          Make sure to use the correct column names from the dataset in the expression. This is what is required in the expression from Pandas docs: The expression to evaluate. This string cannot contain any Python statements, only Python expressions.
          The following operations are supported:
        - Arithmetic operations: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``
        - Boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not)
        - Comparison operators: ``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``
        Furthermore, the following mathematical functions are supported:
        - Trigonometric: ``sin``, ``cos``, ``tan``, ``arcsin``, ``arccos``, \
            ``arctan``, ``arctan2``, ``sinh``, ``cosh``, ``tanh``, ``arcsinh``, \
            ``arccosh`` and ``arctanh``
        - Logarithms: ``log`` natural, ``log10`` base 10, ``log1p`` log(1+x)
        - Absolute Value ``abs``
        - Square root ``sqrt``
        - Exponential ``exp`` and Exponential minus one ``expm1`

Example:
{{
    "ideas": [
        {{
            "name": "debt_to_income_ratio",
            "description": "A feature representing the ratio of debt to income",
            "formula": 'debt/income'
        }}
    ]
}}
"""
