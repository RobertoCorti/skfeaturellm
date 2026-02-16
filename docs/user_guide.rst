User Guide
==========

This guide provides an overview of how to use skfeaturellm for automated feature engineering.


Overview
--------
``skfeaturellm`` uses Large Language Models (LLMs) to suggest meaningful feature transformations for tabular data. The LLM outputs ideas in a structured format (Feature Transformation DSL), which are then validated and executed safely—no ``eval()`` or raw code execution.


Workflow
--------
1. **Fit**: Provide feature descriptions and target description; the LLM generates feature ideas.
2. **Transform**: Execute the generated transformations on your data.
3. **Evaluate** (optional): Assess feature quality with mutual information (classification) or correlation (regression).


LLMFeatureEngineer Parameters
-----------------------------

- **problem_type**: ``"classification"`` or ``"regression"``
- **model_name**: LLM model—any model available from LangChain (e.g., ``"gpt-4o"``, ``"gpt-4"`` for OpenAI; ``"claude-3-5-sonnet"`` for Anthropic; see `LangChain chat models <https://python.langchain.com/docs/integrations/chat/>`_)
- **target_col**: Optional target column name (for future use)
- **max_features**: Maximum number of features to generate
- **feature_prefix**: Prefix for generated feature names (default: ``"llm_feat_"``)
- **kwargs**: Passed to LangChain's ``init_chat_model`` (e.g., ``api_key``, ``temperature``, ``model_provider``)


Feature Transformation DSL
--------------------------
The LLM generates ideas in a structured format with:

- **type**: Transformation type (e.g., ``add``, ``div``, ``log``, ``pow``)
- **feature_name**: Name for the new feature
- **description**: Explanation of the feature
- **columns**: List of column names (1 for unary, 1–2 for binary)
- **parameters**: Optional parameters (e.g., ``{"constant": 2.0}`` for binary ops, ``{"power": 0.5}`` for pow)

Supported transformations:

**Binary** (column-column or column-constant): ``add``, ``sub``, ``mul``, ``div``

**Unary**: ``log``, ``log1p``, ``abs``, ``exp``, ``pow``


API Keys and Provider Configuration
------------------------------------
The library is **model-agnostic**: it works with any LLM provider supported by LangChain (OpenAI, Anthropic, etc.). Set the appropriate API key for your provider (e.g., ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) or pass ``api_key`` and ``model_provider`` to ``LLMFeatureEngineer`` via ``kwargs``. See `LangChain model setup <https://python.langchain.com/docs/integrations/chat/>`_ for provider-specific configuration.
