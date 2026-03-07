from inspect import isclass

from skfeaturellm.exceptions import NotFittedError


def check_is_fitted(estimator, msg=None):
    if isclass(estimator):
        raise TypeError(f"{estimator} is a class, not an instance.")
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator instance.")

    if not _is_fitted(estimator):
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def _is_fitted(estimator):
    fitted_attrs = [
        v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
    ]
    return len(fitted_attrs) > 0
