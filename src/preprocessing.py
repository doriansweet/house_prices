from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ABSENCE_CATEGORICAL_COLUMNS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_NUMERIC_COLUMNS = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]


def to_dense(matrix):
    """Convert a sparse preprocessing result to a dense NumPy array."""

    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def _create_encoder(encoding: str):
    if encoding == "onehot":
        return OneHotEncoder(handle_unknown="ignore")
    if encoding == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    raise ValueError(f"Unknown categorical encoding: {encoding}")


def _numeric_pipeline(strategy: str, scale_numeric: bool) -> Pipeline:
    if strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown numeric imputation strategy: {strategy}")

    steps = [("imputer", imputer)]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _categorical_pipeline(strategy: str, encoding: str) -> Pipeline:
    if strategy == "absent":
        imputer = SimpleImputer(strategy="constant", fill_value="Absent")
    elif strategy == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown categorical imputation strategy: {strategy}")

    return Pipeline(
        [
            ("imputer", imputer),
            ("encoder", _create_encoder(encoding)),
        ]
    )


def create_preprocessor(X, preprocessing_config: dict) -> ColumnTransformer:
    """Create model-specific preprocessing from a config profile."""

    imputation = preprocessing_config["imputation"]
    scale_numeric = preprocessing_config["scale_numeric"]
    encoding = preprocessing_config["categorical_encoding"]

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    if imputation == "simple":
        return ColumnTransformer(
            [
                (
                    "numeric",
                    _numeric_pipeline("median", scale_numeric),
                    numeric_columns,
                ),
                (
                    "categorical",
                    _categorical_pipeline("absent", encoding),
                    categorical_columns,
                ),
            ]
        )

    if imputation != "semantic":
        raise ValueError(f"Unknown imputation mode: {imputation}")

    zero_numeric_columns = [
        column for column in ZERO_NUMERIC_COLUMNS if column in numeric_columns
    ]
    median_numeric_columns = [
        column for column in numeric_columns if column not in zero_numeric_columns
    ]
    absence_categorical_columns = [
        column
        for column in ABSENCE_CATEGORICAL_COLUMNS
        if column in categorical_columns
    ]
    other_categorical_columns = [
        column
        for column in categorical_columns
        if column not in absence_categorical_columns
    ]

    return ColumnTransformer(
        [
            (
                "zero_numeric",
                _numeric_pipeline("zero", scale_numeric),
                zero_numeric_columns,
            ),
            (
                "median_numeric",
                _numeric_pipeline("median", scale_numeric),
                median_numeric_columns,
            ),
            (
                "absence_categorical",
                _categorical_pipeline("absent", encoding),
                absence_categorical_columns,
            ),
            (
                "other_categorical",
                _categorical_pipeline("most_frequent", encoding),
                other_categorical_columns,
            ),
        ]
    )