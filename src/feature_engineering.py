# src/feature_engineering.py

import numpy as np

def add_engineered_features(dataset):
    """
    Adds domain-driven engineered features to the dataset.
    Assumes raw Ames housing columns are present.
    """

    dataset = dataset.copy()  # safety (VERY important)

    dataset['House Age'] = dataset['Yr Sold'] - dataset['Year Built']

    dataset['Year since Remodel'] = (
        dataset['Yr Sold'] - dataset['Year Remod/Add']
    )

    dataset['Is_Remodeled'] = (
        dataset['Year Built'] != dataset['Year Remod/Add']
    ).astype(int)

    dataset['Total Living Area'] = (
        dataset['Gr Liv Area'] + dataset['Total Bsmt SF']
    )

    dataset['Total Finished Area'] = (
        dataset['Gr Liv Area']
        + dataset['BsmtFin SF 1']
        + dataset['BsmtFin SF 2']
    )

    dataset['Has_Basement'] = (
        dataset['Total Bsmt SF'] > 0
    ).astype(int)

    dataset['Total Bathrooms'] = (
        dataset['Bsmt Full Bath']
        + 0.5 * dataset['Bsmt Half Bath']
        + dataset['Full Bath']
        + 0.5 * dataset['Half Bath']
    )

    # Garage logic (ONLY if garage exists)
    dataset['Garage Exists'] = (dataset['Garage Area'] > 0).astype(int)

    dataset['Garage Age'] = np.where(
        dataset['Garage Exists'] == 1,
        dataset['Yr Sold'] - dataset['Garage Yr Blt'],
        0
    )

    dataset['Quality_size_score'] = (
        dataset['Overall Qual'] * dataset['Gr Liv Area']
    )

    dataset['Non_Bedroom_Rooms'] = (
        dataset['TotRms AbvGrd'] - dataset['Bedroom AbvGr']
    )

    return dataset
