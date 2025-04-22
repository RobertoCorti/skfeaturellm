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
``skfeaturellm`` is a Python library that brings the power of Large Language Models (LLMs) to feature engineering for tabular data, wrapped in a familiar scikit-learn–style API. The library aims to leverage LLMs' capabilities to automatically generate and implement meaningful features for your machine learning tasks.


Quickstart
~~~~~~~~~~~~~~~~~~~
The code snippets below are designed to introduce ``skfeaturellm``'s functionality so you can start using its functionality quickly.

Classification
--------------
Example applied to a classification task. The example uses the `Iris plants dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset>`_  from `sklearn.datasets`. The `LLMFeatureEngineer` class is then used to perform feature engineering on the dataset.

.. code-block:: python

    from sklearn.datasets import load_iris
    from skfeaturellm.feature_engineer import LLMFeatureEngineer

    iris_data = load_iris(as_frame=True)
    X, y = iris_data.data, iris_data.target

    target_column = (
        "Classification task predicting species of iris plants "
        "(3 classes: setosa, versicolor, virginica)"
    )
    feature_descriptions = [
        {"name": "sepal length (cm)", "type": "float", "description": "The sepal lengths in centimeters"},
        {"name": "sepal width (cm)", "type": "float", "description": "The sepal widths in centimeters"},
        {"name": "petal length (cm)", "type": "float", "description": "The petal lengths in centimeters"},
        {"name": "petal width (cm)", "type": "float", "description": "The petal widths in centimeters"},
    ]

    llm_feature_engineer = LLMFeatureEngineer(y_col=target_column)

    # Fit the LLMFeatureEngineer
    llm_feature_engineer.fit(X, y, feature_descriptions=feature_descriptions)

    # Transform the data
    X_transformed = llm_feature_engineer.transform(X)

    print(X_transformed.head())
    print(X_transformed.columns)


Regression
-----------
Example applied to a regression task. The example uses the `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_  from `sklearn.datasets`. The `LLMFeatureEngineer` class is then used to perform feature engineering on the dataset.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from skfeaturellm.feature_engineer import LLMFeatureEngineer

    diabetes_data = load_diabetes(as_frame=True)
    X, y = diabetes_data.data, diabetes_data.target

    target_column = "Regression task predicting the quantitative measure of disease progression one year after baselines"
    norm_method = "mean centered and scaled by the standard deviation times the square root of n_samples (i.e. the sum of squares of each column totals 1)"
    feature_descriptions = [
        {"name": "age", "type": "float", "description": f"Age in years ({norm_method})"},
        {"name": "sex", "type": "float", "description": f"Sex of the patient ({norm_method})"},
        {"name": "bmi", "type": "float", "description": f"Body mass index ({norm_method})"},
        {"name": "bp", "type": "float", "description": f"Average Blood pressure ({norm_method})"},
        {"name": "s1", "type": "float", "description": f"TC, total serum cholesterol ({norm_method})"},
        {"name": "s2", "type": "float", "description": f"LDL, low-density lipoprotein ({norm_method})"},
        {"name": "s3", "type": "float", "description": f"HDL, high-density lipoprotein ({norm_method})"},
        {"name": "s4", "type": "float", "description": f"TCH, total cholesterol/HDL ratio ({norm_method})"},
        {"name": "s5", "type": "float", "description": f"s5 ltg, possibly log of serum triglycerides level ({norm_method})"},
        {"name": "s6", "type": "float", "description": f"s6 glu, blood sugar level ({norm_method})"},
    ]

    llm_feature_engineer = LLMFeatureEngineer(y_col=target_column)

    # Fit the LLMFeatureEngineer
    llm_feature_engineer.fit(X, y, feature_descriptions=feature_descriptions)

    # Transform the data
    X_transformed = llm_feature_engineer.transform(X)

    print(X_transformed.head())
    print(X_transformed.columns)
