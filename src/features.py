import numpy as np
import pandas as pd


AVAILABLE_FEATURE_GROUPS = {
    'aggregates',
    'interactions',
    'logs',
    'indicators'
}


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает типы исходных признаков.

    Выполняется всегда, в том числе для baseline.
    Новые признаки не создаёт.
    """

    data = data.copy()

    data['MSSubClass'] = data['MSSubClass'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    return data


def _calculate_total_sf(data: pd.DataFrame) -> pd.Series:
    """Рассчитывает общую площадь дома."""

    return (
        data[
            ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        ]
        .fillna(0)
        .sum(axis=1)
    )


def _calculate_total_porch_sf(
    data: pd.DataFrame
) -> pd.Series:
    """Рассчитывает общую площадь веранд и террас."""

    return (
        data[
            [
                'OpenPorchSF',
                'EnclosedPorch',
                '3SsnPorch',
                'ScreenPorch',
                'WoodDeckSF'
            ]
        ]
        .fillna(0)
        .sum(axis=1)
    )


def add_aggregate_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Добавляет агрегированные признаки площади,
    возраста и количества удобств.
    """

    data = data.copy()

    data['TotalSF'] = _calculate_total_sf(data)

    data['TotalBathrooms'] = (
        data['FullBath'].fillna(0)
        + 0.5 * data['HalfBath'].fillna(0)
        + data['BsmtFullBath'].fillna(0)
        + 0.5 * data['BsmtHalfBath'].fillna(0)
    )

    data['TotalPorchSF'] = (
        _calculate_total_porch_sf(data)
    )

    data['TotalOutdoorSF'] = (
        data['TotalPorchSF']
        + data['PoolArea'].fillna(0)
    )

    data['HouseAge'] = (
        data['YrSold']
        - data['YearBuilt']
    ).clip(lower=0)

    data['RemodAge'] = (
        data['YrSold']
        - data['YearRemodAdd']
    ).clip(lower=0)

    data['GarageAge'] = (
        data['YrSold']
        - data['GarageYrBlt']
    )

    data['GarageAge'] = (
        data['GarageAge']
        .where(data['GarageYrBlt'].notna(), 0)
        .clip(lower=0)
    )

    data['TotalRooms'] = (
        data['TotRmsAbvGrd'].fillna(0)
        + data['FullBath'].fillna(0)
        + data['HalfBath'].fillna(0)
    )

    return data


def add_interaction_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Добавляет произведения, квадраты и отношения.

    Эти признаки особенно полезно проверять
    для линейных моделей.
    """

    data = data.copy()

    if 'TotalSF' in data.columns:
        total_sf = data['TotalSF']
    else:
        total_sf = _calculate_total_sf(data)

    data['OverallQual_GrLivArea'] = (
        data['OverallQual']
        * data['GrLivArea']
    )

    data['OverallQual_TotalSF'] = (
        data['OverallQual']
        * total_sf
    )

    data['OverallQualSquared'] = (
        data['OverallQual'] ** 2
    )

    data['OverallCondSquared'] = (
        data['OverallCond'] ** 2
    )

    data['AverageRoomSize'] = (
        data['GrLivArea']
        / data['TotRmsAbvGrd'].replace(0, np.nan)
    )

    data['GarageCapacity'] = (
        data['GarageCars'].fillna(0)
        * data['GarageArea'].fillna(0)
    )

    data['QualityAgeInteraction'] = (
        data['OverallQual']
        * (
            data['YrSold']
            - data['YearBuilt']
        ).clip(lower=0)
    )

    return data


def add_log_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Добавляет логарифмы числовых признаков
    с правосторонней асимметрией.
    """

    data = data.copy()

    log_columns = [
        'LotArea',
        'GrLivArea',
        '1stFlrSF',
        '2ndFlrSF',
        'TotalBsmtSF',
        'GarageArea',
        'MasVnrArea',
        'WoodDeckSF',
        'OpenPorchSF'
    ]

    for column in log_columns:
        values = (
            data[column]
            .fillna(0)
            .clip(lower=0)
        )

        data[f'{column}Log'] = np.log1p(values)

    if 'TotalSF' in data.columns:
        total_sf = data['TotalSF']
    else:
        total_sf = _calculate_total_sf(data)

    data['TotalSFLog'] = np.log1p(
        total_sf.clip(lower=0)
    )

    return data


def add_indicator_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Добавляет бинарные признаки наличия объектов."""

    data = data.copy()

    data['HasGarage'] = (
        data['GarageArea'].fillna(0) > 0
    ).astype(int)

    data['HasBasement'] = (
        data['TotalBsmtSF'].fillna(0) > 0
    ).astype(int)

    data['HasFireplace'] = (
        data['Fireplaces'].fillna(0) > 0
    ).astype(int)

    data['HasPool'] = (
        data['PoolArea'].fillna(0) > 0
    ).astype(int)

    data['HasSecondFloor'] = (
        data['2ndFlrSF'].fillna(0) > 0
    ).astype(int)

    data['HasMasonryVeneer'] = (
        data['MasVnrArea'].fillna(0) > 0
    ).astype(int)

    data['HasPorch'] = (
        _calculate_total_porch_sf(data) > 0
    ).astype(int)

    data['IsRemodeled'] = (
        data['YearRemodAdd']
        != data['YearBuilt']
    ).astype(int)

    return data


def create_features(
    data: pd.DataFrame,
    groups: list[str] | str
) -> pd.DataFrame:
    """
    Создаёт выбранные группы дополнительных признаков.

    Доступные группы:
    - aggregates;
    - interactions;
    - logs;
    - indicators;
    - all.
    """

    data = data.copy()

    if isinstance(groups, str):
        groups = [groups]

    selected_groups = set(groups)

    if 'all' in selected_groups:
        selected_groups = AVAILABLE_FEATURE_GROUPS.copy()

    unknown_groups = (
        selected_groups
        - AVAILABLE_FEATURE_GROUPS
    )

    if unknown_groups:
        raise ValueError(
            'Неизвестные группы признаков: '
            f'{sorted(unknown_groups)}'
        )

    # Порядок важен: следующие группы могут
    # использовать созданные агрегаты.
    if 'aggregates' in selected_groups:
        data = add_aggregate_features(data)

    if 'interactions' in selected_groups:
        data = add_interaction_features(data)

    if 'logs' in selected_groups:
        data = add_log_features(data)

    if 'indicators' in selected_groups:
        data = add_indicator_features(data)

    return data