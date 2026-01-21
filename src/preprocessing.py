# src/preprocessing.py

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer


# =========================
# COLUMN GROUPS
# =========================

ordinal_cols = [
    'Lot Shape', 'Land Slope', 'Exter Qual', 'Exter Cond',
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
    'BsmtFin Type 1', 'BsmtFin Type 2',
    'Heating QC', 'Kitchen Qual', 'Functional',
    'Fireplace Qu', 'Garage Finish', 'Garage Qual',
    'Garage Cond', 'Pool QC', 'Paved Drive'
]

nominal_low = [
    'MS Zoning', 'Street', 'Alley', 'Land Contour',
    'Utilities', 'Lot Config', 'Condition 1', 'Condition 2',
    'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
    'Mas Vnr Type', 'Foundation', 'Heating', 'Garage Type',
    'Misc Feature', 'Sale Type', 'Sale Condition',
    'Fence', 'Electrical'
]

nominal_high = [
    'Neighborhood', 'Exterior 1st', 'Exterior 2nd'
]

binary_cols = [
    'Central Air', 'Has_Basement',
    'Is_Remodeled', 'Garage Exists'
]

num_cols = [
    'House Age', 'Year since Remodel',
    'Total Living Area', 'Total Finished Area',
    'Total Bathrooms', 'Garage Age',
    'Quality_size_score', 'Non_Bedroom_Rooms'
]


# =========================
# ORDINAL ORDERS
# =========================

Qual_bsd_fea = [
    'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond',
    'Heating QC', 'Kitchen Qual', 'Fireplace Qu',
    'Garage Qual', 'Garage Cond', 'Pool QC'
]
qual_order = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

Bsmt_expos = ['Bsmt Exposure']
bsmt_exposure_order = ['None', 'No', 'Mn', 'Av', 'Gd']

BsmtFinish = ['BsmtFin Type 1', 'BsmtFin Type 2']
bsmt_finish_order = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']

GarageFinsh = ['Garage Finish']
garage_finish_order = ['None', 'Unf', 'RFn', 'Fin']

Functional_col = ['Functional']
functional_order = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']

pvd_drive = ['Paved Drive']
paved_drive_order = ['N', 'P', 'Y']

land_slope_col = ['Land Slope']
land_slope_order = ['Sev', 'Mod', 'Gtl']

lot_shape_col = ['Lot Shape']
lot_shape_order = ['IR3', 'IR2', 'IR1', 'Reg']


ordinal_columns = (
    Qual_bsd_fea
    + Bsmt_expos
    + BsmtFinish
    + GarageFinsh
    + Functional_col
    + pvd_drive
    + land_slope_col
    + lot_shape_col
)

ordinal_orders = (
    [qual_order] * len(Qual_bsd_fea)
    + [bsmt_exposure_order] * len(Bsmt_expos)
    + [bsmt_finish_order] * len(BsmtFinish)
    + [garage_finish_order] * len(GarageFinsh)
    + [functional_order] * len(Functional_col)
    + [paved_drive_order] * len(pvd_drive)
    + [land_slope_order] * len(land_slope_col)
    + [lot_shape_order] * len(lot_shape_col)
)

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

# =========================
# TRANSFORMERS
# =========================

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ord_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(
        categories=ordinal_orders,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

nom_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])

bin_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])


# =========================
# MAIN FUNCTION
# =========================

def get_preprocessor():
    """
    Returns a ColumnTransformer with all preprocessing steps.
    """
    return ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('ord', ord_transformer, ordinal_columns),
            ('nom_low', nom_transformer, nominal_low),
            ('nom_high', nom_transformer, nominal_high),
            ('bin', bin_transformer, binary_cols)
        ]
    )
