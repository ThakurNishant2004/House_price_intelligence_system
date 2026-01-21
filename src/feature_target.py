import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

def add_log_target(dataset, target_col='SalePrice'):
    """
    Applies log1p transformation to the target variable.
    
    Parameters:
    - dataset : pandas DataFrame
    - target_col : column name of target variable
    
    Returns:
    - dataset with new column: <target_col>_log
    """
    dataset = dataset.copy()
    dataset[f'{target_col}_log'] = np.log1p(dataset[target_col])
    return dataset