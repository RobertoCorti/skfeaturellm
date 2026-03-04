Examples
========

This page contains practical examples of using ``skfeaturellm`` in different scenarios.


Classification
--------------
Example applied to a classification task using the `Iris plants dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset>`_ from ``sklearn.datasets``. Passing ``y`` to ``fit()`` injects dataset statistics into the LLM prompt for richer feature suggestions.

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

    engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        max_features=5,
    )

    # Fit on training data only — passing y injects dataset statistics into the prompt
    engineer.fit(
        X=X_train,
        y=y_train,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    X_train_transformed = engineer.transform(X_train)
    X_test_transformed = engineer.transform(X_test)

    print(X_train_transformed.columns.tolist())


Regression
-----------
Example applied to a regression task using the `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_ from ``sklearn.datasets``.

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

    engineer = LLMFeatureEngineer(
        problem_type="regression",
        model_name="gpt-4o",
        max_features=5,
    )

    engineer.fit(
        X=X_train,
        y=y_train,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    X_train_transformed = engineer.transform(X_train)
    X_test_transformed = engineer.transform(X_test)


Feature Evaluation and Selection
---------------------------------
``evaluate_features()`` scores each generated feature using mutual information (classification) or Pearson/Spearman correlation (regression). Use the results to select only features that provide real signal before training your final model.

.. code-block:: python

    # Score features on training data
    eval_result = engineer.evaluate_features(X_train, y_train, is_transformed=False)
    print(eval_result.summary())

    # Select features with a positive mutual information score
    scores = eval_result.summary()
    good_features = scores[scores["mutual_information"] > 0].index.tolist()

    # Build train and test sets with only the selected features
    base_cols = X_train.columns.tolist()
    X_train_eng = engineer.transform(X_train)[base_cols + good_features]
    X_test_eng = engineer.transform(X_test)[base_cols + good_features]


Production Pipeline
--------------------
After evaluating and selecting features, export them to a ``FeatureEngineeringTransformer`` for use in a scikit-learn ``Pipeline``. This separates the LLM exploration phase from deterministic production inference.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from skfeaturellm import LLMFeatureEngineer, FeatureEngineeringTransformer

    # --- Exploration: fit and evaluate ---
    engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        max_features=10,
    )
    engineer.fit(X_train, y=y_train, feature_descriptions=feature_descriptions)
    engineer.transform(X_train)  # populates generated_features

    # Evaluate and select
    eval_result = engineer.evaluate_features(X_train, y_train)
    good_features = (
        eval_result.summary()[eval_result.summary()["mutual_information"] > 0]
        .index.tolist()
    )

    # --- Production: export to a deterministic transformer ---
    transformer = engineer.to_transformer(features=good_features)

    pipeline = Pipeline([
        ("features", transformer),
        ("model", XGBClassifier()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)


Saving and Loading
------------------
Serialize the ``FeatureEngineeringTransformer`` to JSON so the LLM is never called again. Only the transformation configs are stored — call ``fit()`` to re-learn stateful parameters (e.g., bin edges) on any data split.

.. code-block:: python

    from skfeaturellm import FeatureEngineeringTransformer

    # Save transformation configs
    transformer.save("transformer.json")

    # Restore in a later session or on a different machine
    loaded = FeatureEngineeringTransformer.load("transformer.json")

    # Fit on training data and apply — no LLM call required
    pipeline = Pipeline([("features", loaded), ("model", XGBClassifier())])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)


Notebook Tutorial
-----------------
A complete end-to-end tutorial using the `Bank Loan Credit Risk dataset <https://www.kaggle.com/datasets/udaymalviya/bank-loan-data>`_ is available as a Jupyter notebook in the ``examples/`` directory of the repository:

- `01_SKFeatureLLM_Tutorial.ipynb <https://github.com/RobertoCorti/skfeaturellm/blob/main/examples/01_SKFeatureLLM_Tutorial.ipynb>`_

The notebook covers: data loading with ``kagglehub``, baseline XGBoost, LLM feature engineering with dataset statistics injection, per-feature evaluation, feature selection, and production deployment with ``FeatureEngineeringTransformer`` inside a scikit-learn ``Pipeline``.
