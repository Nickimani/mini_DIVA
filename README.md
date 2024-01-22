# impute-tools

This toolbox combines multiple imputation methods into a single workflow, to evaluate imputation performance and its impact on downstream supervised learning problems.

Current features are outlined below.

## Imputation methods
* Naive
  * Zero, mean, median, mode (default to mode for categorical variables)
    
* Statistical
  
  These methods use a pre-specified distance metric
  * MICE (sklearn's IterativeImputer)
  * customMICE, with support for specifying regressors as 'linear', 'lasso', 'ridge' (numerical variables) and 'logistic', 'random-forest' (categorical variables)
  

## Datasets
The following datasets have been obtained from the open-source University California Irvine repository. Datasets have been harmonized as .csv files with features identified as named columns.
  * Air Quality
  * Automobile
  * Beijing Multi-site
    * Contains per-hour measurements of air pollutants taken from a dozen air-quality observation locations, sourced from the Beijing Municipal Environmental Monitoring Center. Each of the air-quality sites has corresponding meteorological data, associated with the closest weather station, as provided by the China Meteorological Administration. The observation locations are comprised of 8 urban, 3 suburban, and 1 remote background station. The dataset spans from the beginning of March 2013 to the end of February 2017.
    * Variables: 18 (15 numerical, 3 categorical).
    * 33.3 Mb; 382,168 records.
  * Credit Approval
     * Records of credit application decisions. All attribute names and categories have been encoded to ensure confidentiality of the data. The decision to approve or decline the application is encoded as +/-.
     * Variables: 15 (6 numerical, 9 categorical).
     * 31 Kb; 653 records.
  * Dermatology
  * Iris
  * Monks
     * An artificial dataset with exclusively categorical features. There are 3 MONK's datasets available from UCI, each partitioned into training and testing. We include here only 'Monks-1-test'
     * Variables: 7 (0 numerical, 7 categorical).
     * 10 Kb; 432 records.
  * Soybean
  * Wine Quality (red, white)
     * These two datasets contain physicochemical measurements of red and white wines, along with a quality integer score between 0 and 10. The properties measured are: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality.
     * Variables: 12 (12 numerical, 0 categorical).
     * 123 Kb, 382 Kb; 1,599 records, 4,898 records (red and white, respectively).
  * Yeast

## Pipeline
The imputation methods are incorporated into a single pipeline, with functionality for cross-validation and encoding of categorical variables (either 'label' or 'one-hot' encoding). The pipeline may be run with any dataset, specifying the fraction(s) of missingness and the repeats. From this, both the imputation error (either MSE, misclassication rate, or 'mix' error) is calculated along with the performance of a downstream task (either RF-classifier or RF-regressor).
