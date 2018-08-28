# Changelog

## [0.1.1][2017-12]

### Added

- transformers: `OneHotEncoder`

### Changed

- transformers: `Preprocesser`
	- Functionality for detecting collinearity amongst categorical features using Cramer's V statistics.
	- Reimplement `OneHotEncoder` allowing user to drop the first level of a categorical feature.
	- Added arguments to specify whether to output pandas DataFrame or numpy ndarray and whether to use one hot encoding for the categorical columns.

## [0.0.1][2017-11]

### Added

- resamplers: `RandomUndersampler`.
- explainers: `PartialDependenceExplainer`.
- visualization: `vis_importance`, `vis_coef`.
- transformers: `BoxTransformer`, `MultipleImputer`, `Preprocesser`, `ColumnExtractor`.
