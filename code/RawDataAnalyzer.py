import pandas as pd



class RawDataAnalyzer:
    """
    A class for analyzing and cleaning the initial data.
    """
    def __init__(self):
        """
        Initializes the RawDataAnalyzer class.
        """
        
    def analyze(self, data: pd.DataFrame):
        """
        Analyzes the cleaned data by providing insights such as missing values and dataset shape.

        Args:
            data (pd.DataFrame): The cleaned data to analyze.
        """
        print(f"\nMissing Values:\n{data.isnull().sum()}")
        print(f"\nDataset shape: {data.shape}")

    def clean_data(self, data: pd.DataFrame, columns_to_clean: list[str]):
        """
        Cleans the raw data by removing duplicates and handling missing values.

        Args:
            data (pd.DataFrame): The raw data to clean.
            columns_to_clean (list[str]): The list of columns to check for missing values.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        cleaned_data = data.drop_duplicates(inplace=False)
        cleaned_data[columns_to_clean] = (
            cleaned_data[columns_to_clean].replace(r'^\s*$', pd.NA, regex=True)
            )
        cleaned_data = cleaned_data.dropna(subset=columns_to_clean, inplace=False)
        return cleaned_data

