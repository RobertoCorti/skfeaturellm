"""
Transformation executor for applying feature transformations to DataFrames.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

import pandas as pd

from skfeaturellm.transformations.base import BaseTransformation, TransformationError


class TransformationParseError(TransformationError):
    """Raised when parsing a transformation definition fails."""

    pass


_TRANSFORMATION_REGISTRY: Dict[str, Type[BaseTransformation]] = {}


def register_transformation(name: str):
    """
    Decorator to register a transformation class with a type name.

    Parameters
    ----------
    name : str
        The type name used in JSON/YAML configs (e.g., "add", "div")
    """

    def decorator(cls: Type[BaseTransformation]) -> Type[BaseTransformation]:
        _TRANSFORMATION_REGISTRY[name] = cls
        return cls

    return decorator


def get_registered_transformations() -> Dict[str, Type[BaseTransformation]]:
    """Return a copy of the transformation registry."""
    return _TRANSFORMATION_REGISTRY.copy()


def get_transformation_types_for_prompt() -> str:
    """
    Generate documentation of available transformation types for LLM prompts.

    This function dynamically generates the transformation types section
    by querying the registry and calling get_prompt_description() on each
    registered transformation class.

    Returns
    -------
    str
        Formatted documentation string listing all available transformations
    """
    lines = []
    for name, cls in sorted(_TRANSFORMATION_REGISTRY.items()):
        description = cls.get_prompt_description()
        lines.append(f'- "{name}": {description}')

    return "\n".join(lines)


def get_unary_operation_types() -> Set[str]:
    """
    Get the set of registered unary operation type names.

    Returns
    -------
    Set[str]
        Set of unary operation names (e.g., {"log", "sqrt", "abs"})
    """
    # Import here to avoid circular imports
    from skfeaturellm.transformations.unary.arithmetic import UnaryTransformation

    unary_ops = set()
    for name, cls in _TRANSFORMATION_REGISTRY.items():
        if issubclass(cls, UnaryTransformation):
            unary_ops.add(name)
    return unary_ops


def get_binary_operation_types() -> Set[str]:
    """
    Get the set of registered binary operation type names.

    Returns
    -------
    Set[str]
        Set of binary operation names (e.g., {"add", "sub", "mul", "div"})
    """
    # Import here to avoid circular imports
    from skfeaturellm.transformations.binary.arithmetic import (
        BinaryArithmeticTransformation,
    )

    binary_ops = set()
    for name, cls in _TRANSFORMATION_REGISTRY.items():
        if issubclass(cls, BinaryArithmeticTransformation):
            binary_ops.add(name)
    return binary_ops


def get_all_operation_types() -> Set[str]:
    """
    Get all registered operation type names.

    Returns
    -------
    Set[str]
        Set of all operation names
    """
    return set(_TRANSFORMATION_REGISTRY.keys())


class TransformationExecutor:
    """
    Executes a set of transformations against a DataFrame.

    The executor can be initialized with transformations directly, or loaded
    from JSON/YAML configuration files.

    Parameters
    ----------
    transformations : List[BaseTransformation], optional
        List of transformation objects to execute
    raise_on_error : bool, default=True
        If True, raise exceptions on transformation errors.
        If False, skip failed transformations with a warning.

    Examples
    --------
    Direct instantiation:

    >>> from skfeaturellm.transformations import AddTransformation, DivTransformation
    >>> executor = TransformationExecutor(transformations=[
    ...     DivTransformation("ratio", "a", right_column="b"),
    ...     AddTransformation("sum", "a", right_column="b"),
    ... ])
    >>> result_df = executor.execute(df)

    From JSON file:

    >>> executor = TransformationExecutor.from_json("transformations.json")
    >>> result_df = executor.execute(df)

    From dict (e.g., LLM output):

    >>> config = {"transformations": [{"type": "add", "feature_name": "sum", ...}]}
    >>> executor = TransformationExecutor.from_dict(config)
    >>> result_df = executor.execute(df)
    """

    def __init__(
        self,
        transformations: Optional[List[BaseTransformation]] = None,
        raise_on_error: bool = True,
    ):
        self.transformations = transformations or []
        self.raise_on_error = raise_on_error

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        raise_on_error: bool = True,
    ) -> "TransformationExecutor":
        """
        Create an executor from a dictionary configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dict with a "transformations" key containing
            a list of transformation definitions
        raise_on_error : bool, default=True
            If True, raise exceptions on transformation errors

        Returns
        -------
        TransformationExecutor
            Configured executor instance

        Raises
        ------
        TransformationParseError
            If the configuration is invalid
        """
        if "transformations" not in config:
            raise TransformationParseError(
                "Configuration must contain a 'transformations' key"
            )

        transformations = []
        for i, t_config in enumerate(config["transformations"]):
            transformation = cls._parse_transformation(t_config, index=i)
            transformations.append(transformation)

        return cls(transformations=transformations, raise_on_error=raise_on_error)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        raise_on_error: bool = True,
    ) -> "TransformationExecutor":
        """
        Create an executor from a JSON configuration file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the JSON configuration file
        raise_on_error : bool, default=True
            If True, raise exceptions on transformation errors

        Returns
        -------
        TransformationExecutor
            Configured executor instance
        """
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        return cls.from_dict(config, raise_on_error=raise_on_error)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        raise_on_error: bool = True,
    ) -> "TransformationExecutor":
        """
        Create an executor from a YAML configuration file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML configuration file
        raise_on_error : bool, default=True
            If True, raise exceptions on transformation errors

        Returns
        -------
        TransformationExecutor
            Configured executor instance

        Raises
        ------
        ImportError
            If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config, raise_on_error=raise_on_error)

    @classmethod
    def _parse_transformation(
        cls, config: Dict[str, Any], index: int
    ) -> BaseTransformation:
        """
        Parse a single transformation definition into a transformation object.

        Parameters
        ----------
        config : Dict[str, Any]
            Transformation configuration dict
        index : int
            Index of the transformation (for error messages)

        Returns
        -------
        BaseTransformation
            The parsed transformation object
        """
        if "type" not in config:
            raise TransformationParseError(
                f"Transformation at index {index} missing required 'type' field"
            )

        t_type = config["type"]
        if t_type not in _TRANSFORMATION_REGISTRY:
            available = list(_TRANSFORMATION_REGISTRY.keys())
            raise TransformationParseError(
                f"Unknown transformation type '{t_type}' at index {index}. "
                f"Available types: {available}"
            )

        t_class = _TRANSFORMATION_REGISTRY[t_type]

        kwargs = {k: v for k, v in config.items() if k != "type"}

        try:
            return t_class(**kwargs)
        except TypeError as e:
            raise TransformationParseError(
                f"Invalid arguments for transformation '{t_type}' at index {index}: {e}"
            )
        except ValueError as e:
            raise TransformationParseError(e)

    def execute(
        self,
        df: pd.DataFrame,
        transformations: Optional[List[BaseTransformation]] = None,
    ) -> pd.DataFrame:
        """
        Execute all transformations and return a DataFrame with new features.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame
        transformations : List[BaseTransformation], optional
            List of transformations to execute. If not provided, uses
            self.transformations.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with new feature columns added

        Raises
        ------
        TransformationError
            If a transformation fails and raise_on_error is True
        """
        # Use provided transformations or fall back to instance transformations
        transforms = (
            transformations if transformations is not None else self.transformations
        )

        if not transforms:
            warnings.warn("No transformations to execute.")
            return df.copy()

        result_df = df.copy()

        for transformation in transforms:
            try:
                feature_series = transformation.execute(df)
                result_df[transformation.feature_name] = feature_series
            except TransformationError as e:
                if self.raise_on_error:
                    raise
                warnings.warn(
                    f"Transformation '{transformation.feature_name}' failed: {e}. "
                    f"Skipping."
                )

        return result_df

    def get_required_columns(
        self, transformations: Optional[List[BaseTransformation]] = None
    ) -> Set[str]:
        """
        Get all column names required by transformations.

        Parameters
        ----------
        transformations : List[BaseTransformation], optional
            List of transformations to analyze. If not provided, uses
            self.transformations.

        Returns
        -------
        Set[str]
            Set of required column names
        """
        transforms = (
            transformations if transformations is not None else self.transformations
        )

        required: Set[str] = set()
        for transformation in transforms:
            required.update(transformation.get_required_columns())
        return required
