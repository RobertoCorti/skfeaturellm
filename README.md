# SKFeatureLLM

<div align="center">
  <img src="docs/_static/logo.png" alt="SKFeatureLLM Logo" width="200"/>
</div>

<div align="center">
  <a href="https://skfeaturellm.readthedocs.io/">
    <img src="https://readthedocs.org/projects/skfeaturellm/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://pypi.org/project/skfeaturellm/">
    <img src="https://img.shields.io/pypi/v/skfeaturellm" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/skfeaturellm/">
    <img src="https://img.shields.io/pypi/pyversions/skfeaturellm" alt="Python Versions">
  </a>
  <a href="https://github.com/yourusername/skfeaturellm/actions">
    <img src="https://github.com/yourusername/skfeaturellm/actions/workflows/tests.yml/badge.svg" alt="Tests Status">
  </a>
  <a href="https://codecov.io/gh/yourusername/skfeaturellm">
    <img src="https://codecov.io/gh/yourusername/skfeaturellm/branch/main/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://github.com/yourusername/skfeaturellm/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
</div>

SKFeatureLLM is a Python library that brings the power of Large Language Models (LLMs) to feature engineering for tabular data, wrapped in a familiar scikit-learn–style API. The library aims to leverage LLMs' capabilities to automatically generate and implement meaningful features for your machine learning tasks.

## 📑 Table of Contents

- [🌟 Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📚 Documentation](#-documentation)
- [💡 Examples](#-examples)
- [🛠 Development](#-development)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🌟 Key Features

- 🤖 LLM-powered feature engineering
- 🔌 Model-agnostic: works with any LLM provider (OpenAI, Anthropic, etc.)
- 🛠 Scikit-learn compatible API
- 📊 Comprehensive feature evaluation and reporting
- 🎯 Support for both supervised and unsupervised feature engineering


## 🛠 Development

1. Clone the repository
```bash
git clone https://github.com/yourusername/skfeaturellm.git
cd skfeaturellm
```

2. Install dependencies
```bash
poetry install
```

3. Activate the virtual environment
```bash
poetry env use python3 && poetry install && source $(poetry env info --path)/bin/activate
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: If you find a bug, please open an issue with a detailed description.
2. **Suggest Features**: Have an idea for a new feature? Open an issue to discuss it.
3. **Submit Pull Requests**: We love PRs! Here's how to submit one:
   - Fork the repository
   - Create a new branch for your feature
   - Make your changes
   - Submit a pull request

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skfeaturellm.git
cd skfeaturellm
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest
```

4. Format code:
```bash
black .
isort .
```

### Code Style

We use:
- Black for code formatting
- isort for import sorting
- pylint for linting
- mypy for type checking

Please ensure your code passes all checks before submitting a PR.

## 👤 Author

- **Roberto Corti** - [GitHub](https://github.com/RobertoCorti)
- **Stefano Polo** - [GitHub](https://github.com/stefano-polo)

For more examples and detailed documentation, visit our [documentation site](https://skfeaturellm.readthedocs.io/).
