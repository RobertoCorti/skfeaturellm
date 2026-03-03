Get Started
===============

The following information is designed to get users up and running with ``skfeaturellm`` quickly. For more detailed information, see the links in each of the subsections.

Installation
~~~~~~~~~~~~~~~~~
``skfeaturellm`` currently supports:

- environments with python version 3.10, 3.11, or 3.12.
- operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher
- installation via `PyPI <https://pypi.org/project/skfeaturellm/>`_

Please see the :doc:`installation` guide for step-by-step instructions on the package installation.


Key Concepts
~~~~~~~~~~~~~~~~~~~
``skfeaturellm`` is a Python library that brings the power of Large Language Models (LLMs) to feature engineering for tabular data, wrapped in a familiar scikit-learn–style API. The library is **model-agnostic**: it works with any LLM available from LangChain (OpenAI, Anthropic, etc.). It leverages LLMs' capabilities to automatically generate and implement meaningful features for your machine learning tasks.


Quickstart
~~~~~~~~~~~~~~~~~~~
The code snippets below introduce ``skfeaturellm``'s core workflow. Both examples follow the same pattern:

1. Split data into train and test sets.
2. Call ``fit()`` on the training set only, passing ``y`` so that dataset statistics are injected into the LLM prompt.
3. Call ``transform()`` on each split independently.

.. note::
   Always fit on training data only to avoid leaking test-set information into the LLM prompt.

Classification
--------------
Example applied to a classification task. The example uses the `Iris plants dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset>`_ from `sklearn.datasets`.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from skfeaturellm import LLMFeatureEngineer

    iris_data = load_iris(as_frame=True)
    X, y = iris_data.data, iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_description = (
        "Classification task predicting species of iris plants "
        "(3 classes: setosa, versicolor, virginica)"
    )
    feature_descriptions = [
        {"name": "sepal length (cm)", "type": "float64", "description": "The sepal lengths in centimeters"},
        {"name": "sepal width (cm)", "type": "float64", "description": "The sepal widths in centimeters"},
        {"name": "petal length (cm)", "type": "float64", "description": "The petal lengths in centimeters"},
        {"name": "petal width (cm)", "type": "float64", "description": "The petal widths in centimeters"},
    ]

    llm_feature_engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        max_features=5,
    )

    # Fit on training data — passing y injects dataset statistics into the LLM prompt
    llm_feature_engineer.fit(
        X=X_train,
        y=y_train,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    # Transform train and test independently
    X_train_transformed = llm_feature_engineer.transform(X_train)
    X_test_transformed = llm_feature_engineer.transform(X_test)

    print(X_train_transformed.columns.tolist())


Regression
-----------
Example applied to a regression task. The example uses the `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_ from `sklearn.datasets`.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from skfeaturellm import LLMFeatureEngineer

    diabetes_data = load_diabetes(as_frame=True)
    X, y = diabetes_data.data, diabetes_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_description = (
        "Regression task predicting the quantitative measure of disease progression "
        "one year after baseline"
    )
    norm_method = "mean centered and scaled by the standard deviation"
    feature_descriptions = [
        {"name": "age", "type": "float64", "description": f"Age in years ({norm_method})"},
        {"name": "sex", "type": "float64", "description": f"Sex of the patient ({norm_method})"},
        {"name": "bmi", "type": "float64", "description": f"Body mass index ({norm_method})"},
        {"name": "bp", "type": "float64", "description": f"Average blood pressure ({norm_method})"},
        {"name": "s1", "type": "float64", "description": f"TC, total serum cholesterol ({norm_method})"},
        {"name": "s2", "type": "float64", "description": f"LDL, low-density lipoprotein ({norm_method})"},
        {"name": "s3", "type": "float64", "description": f"HDL, high-density lipoprotein ({norm_method})"},
        {"name": "s4", "type": "float64", "description": f"TCH, total cholesterol/HDL ratio ({norm_method})"},
        {"name": "s5", "type": "float64", "description": f"s5 ltg, possibly log of serum triglycerides ({norm_method})"},
        {"name": "s6", "type": "float64", "description": f"s6 glu, blood sugar level ({norm_method})"},
    ]

    llm_feature_engineer = LLMFeatureEngineer(
        problem_type="regression",
        model_name="gpt-4o",
        max_features=5,
    )

    # Fit on training data — passing y injects dataset statistics into the LLM prompt
    llm_feature_engineer.fit(
        X=X_train,
        y=y_train,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    # Transform train and test independently
    X_train_transformed = llm_feature_engineer.transform(X_train)
    X_test_transformed = llm_feature_engineer.transform(X_test)

    print(X_train_transformed.columns.tolist())
