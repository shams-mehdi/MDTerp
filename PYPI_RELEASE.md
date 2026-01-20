# PyPI Release Guide for MDTerp v1.5.0

This guide provides step-by-step instructions for building and publishing MDTerp to PyPI.

## Prerequisites

1. **Install build tools**:
   ```bash
   pip install --upgrade build twine
   ```

2. **PyPI account**: Ensure you have accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

3. **API tokens**: Create API tokens for authentication:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

## Pre-Release Checklist

- [x] Version number updated in `pyproject.toml` (1.5.0)
- [x] Version number updated in `MDTerp/__init__.py` (1.5.0)
- [x] Changelog updated in `docs/changelog.md`
- [x] README.md updated with new features
- [x] All new modules added to `__init__.py`
- [x] All tests passing (run `pytest` if available)
- [x] Git tag created for release

## Build the Package

1. **Clean previous builds**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. **Build source distribution and wheel**:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/MDTerp-1.5.0.tar.gz` (source distribution)
   - `dist/MDTerp-1.5.0-py3-none-any.whl` (wheel)

3. **Verify the build**:
   ```bash
   ls -lh dist/
   ```

   You should see both files listed.

## Test on TestPyPI (Recommended)

1. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

   Enter your TestPyPI API token when prompted.

2. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps MDTerp
   ```

3. **Verify the installation**:
   ```python
   import MDTerp
   print(MDTerp.__version__)  # Should print: 1.5.0
   from MDTerp import visualization, analysis
   print("All modules imported successfully!")
   ```

## Publish to PyPI (Production)

1. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

   Enter your PyPI API token when prompted.

2. **Verify on PyPI**:
   - Visit: https://pypi.org/project/MDTerp/
   - Check version is 1.5.0
   - Verify README displays correctly

3. **Test production installation**:
   ```bash
   pip install --upgrade MDTerp
   ```

4. **Verify the installation**:
   ```python
   import MDTerp
   print(MDTerp.__version__)  # Should print: 1.5.0

   # Test new features
   from MDTerp import visualization, analysis
   print("Visualization module:", dir(visualization))
   print("Analysis module:", dir(analysis))
   ```

## Create GitHub Release

1. **Create a git tag**:
   ```bash
   git tag -a v1.5.0 -m "Release version 1.5.0"
   git push origin v1.5.0
   ```

2. **Create GitHub release**:
   - Go to: https://github.com/shams-mehdi/MDTerp/releases/new
   - Tag: v1.5.0
   - Title: MDTerp v1.5.0 - Multi-CPU, Visualization, and Analysis
   - Description: Copy from `docs/changelog.md`

## Post-Release

1. **Update documentation site** (if applicable):
   ```bash
   mkdocs gh-deploy
   ```

2. **Announce the release**:
   - Update project README if needed
   - Notify users/collaborators
   - Tweet/post about new features

## Troubleshooting

### Build Failures

If `python -m build` fails:
- Check `pyproject.toml` syntax
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### Upload Failures

If upload fails:
- Check API token is valid
- Verify package name isn't taken
- Ensure version number is unique (can't re-upload same version)

### Import Errors After Installation

If imports fail after installation:
- Check `__init__.py` exports all modules
- Verify module files exist in package
- Check for circular imports

## Version Information

- **Current Version**: 1.5.0
- **Python Requirement**: >=3.8
- **Key Dependencies**: numpy, scikit-learn, scipy, matplotlib

## Changes in v1.5.0

This release includes:

1. **Bug Fix**: Fixed critical multiprocessing pickling bug
2. **Feature 5**: Comprehensive visualization and analysis utilities
3. **Feature 8**: Package ready for PyPI update

For full changelog, see `docs/changelog.md`.

## Contact

- **Author**: Shams Mehdi
- **Email**: shamsmehdi222@gmail.com
- **GitHub**: https://github.com/shams-mehdi/MDTerp
- **Issues**: https://github.com/shams-mehdi/MDTerp/issues
