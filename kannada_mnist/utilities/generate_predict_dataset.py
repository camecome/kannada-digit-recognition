from pathlib import Path

import pandas as pd

from kannada_mnist.utilities.constants import DEFAULT_DATA_DIR, PIXEL_MAX_VALUE


def generate_predict_dataset(
    fraction: float = 0.2,
    random_seed: int = 42,
):
    test_csv_path = Path(DEFAULT_DATA_DIR) / "test.csv"
    predict_csv_path = Path(DEFAULT_DATA_DIR) / "predict.csv"

    if not test_csv_path.exists():
        raise FileNotFoundError(f"{test_csv_path} does not exist!")

    df = pd.read_csv(test_csv_path)
    df_sample = df.sample(frac=fraction, random_state=random_seed)
    df_sample = df_sample.drop(columns=df_sample.columns[0])

    df_sample *= PIXEL_MAX_VALUE

    df_sample.to_csv(predict_csv_path, index=False)
    print(f"Saved predict dataset to {predict_csv_path}, shape: {df_sample.shape}")
