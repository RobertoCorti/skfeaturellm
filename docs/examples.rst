Examples
========

This page contains practical examples of using ``skfeaturellm`` in different scenarios.


Classification
--------------
Example applied to a classification task using the `Iris plants dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset>`_ from ``sklearn.datasets``. The ``LLMFeatureEngineer`` uses an LLM to generate feature ideas (e.g., ratios, log transforms) and then executes them via the Feature Transformation DSL.

.. code-block:: python

    from sklearn.datasets import load_iris
    from skfeaturellm import LLMFeatureEngineer

    iris_data = load_iris(as_frame=True)
    X, y = iris_data.data, iris_data.target

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

    llm_feature_engineer.fit(
        X=X,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    X_transformed = llm_feature_engineer.transform(X)
    print(X_transformed.head())
    print(X_transformed.columns)


Regression
-----------
Example applied to a regression task using the `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_ from ``sklearn.datasets``.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from skfeaturellm import LLMFeatureEngineer

    diabetes_data = load_diabetes(as_frame=True)
    X, y = diabetes_data.data, diabetes_data.target

    target_description = (
        "Regression task predicting the quantitative measure of disease progression "
        "one year after baselines"
    )
    norm_method = "mean centered and scaled by the standard deviation"
    feature_descriptions = [
        {"name": "age", "type": "float64", "description": f"Age in years ({norm_method})"},
        {"name": "sex", "type": "float64", "description": f"Sex of the patient ({norm_method})"},
        {"name": "bmi", "type": "float64", "description": f"Body mass index ({norm_method})"},
        {"name": "bp", "type": "float64", "description": f"Average blood pressure ({norm_method})"},
        {"name": "s1", "type": "float64", "description": f"TC, total serum cholesterol ({norm_method})"},
        {"name": "s2", "type": "float64", "description": f"LDL ({norm_method})"},
        {"name": "s3", "type": "float64", "description": f"HDL ({norm_method})"},
        {"name": "s4", "type": "float64", "description": f"TCH ratio ({norm_method})"},
        {"name": "s5", "type": "float64", "description": f"s5 ltg, log serum triglycerides ({norm_method})"},
        {"name": "s6", "type": "float64", "description": f"s6 glu, blood sugar ({norm_method})"},
    ]

    llm_feature_engineer = LLMFeatureEngineer(
        problem_type="regression",
        model_name="gpt-4o",
        max_features=5,
    )

    llm_feature_engineer.fit(
        X=X,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    X_transformed = llm_feature_engineer.transform(X)
    print(X_transformed.head())
    print(X_transformed.columns)


Feature Evaluation
------------------
After fitting and transforming, you can evaluate the quality of generated features using mutual information (classification) or correlation (regression):

.. code-block:: python

    # After fit and transform
    eval_results = llm_feature_engineer.evaluate_features(X, y, is_transformed=False)
    print(eval_results)
