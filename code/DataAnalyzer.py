import pandas as pd
import logging

logging.basicConfig(
    filename="./logs/database.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)



class DataAnalyzer:
    """
    A class to analyze IoT telemetry data.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataAnalyzer with a pandas DataFrame.
        
        Params: 
        data (pd.DataFrame): A pandas DataFrame containing the IoT telemetry data.
        """
        self.data = data
        logger.info("Analyzer initialized: %s records", len(data))

    def check_missing_values(self):
        """
        Check for missing values in the dataset and print the count of missing values for each column.
        """
        missing_values = self.data.isnull().sum()
        logger.info("Missing values checked")
        print("Missing values in each column:")
        print(missing_values)

    def handle_missing_values(self, strategy: str = "drop"):
        """
        Handle missing values in the dataset based on the specified strategy.
        
        Params:
        strategy (str): The strategy to handle missing values. Options are "drop" or "fill".
                        - "drop": Drop rows with any missing values.
                        - "fill": Fill missing values with the mean of the column.
        """
        initial_data_length = len(self.data)
        try:
            if strategy == "drop":
                self.data = self.data.dropna()
                logger.info("Missing values removed: %s records", initial_data_length - len(self.data))
                print("Dropped rows with missing values.")
            elif strategy == "fill":
                for column in self.data.columns:
                    if self.data[column].dtype in ["int64", "float64"]:
                        self.data[column] = self.data[column].fillna(
                            self.data[column].mean()
                        )
                    else:
                        self.data[column] = self.data[column].fillna(
                            self.data[column].mode()[0]
                        )
                logger.info("Missing values filled using mean for numeric and mode for categorical columns")
                print("Filled missing values using column mean/mode.")
            else:
                logger.warning("Invalid missing values strategy: %s", strategy)
                raise ValueError("Invalid strategy. Use 'drop' or 'fill'.")
        except Exception as e:
            logger.exception("Missing values handling failed: %s", e)
            raise

    def check_data_types(self):
        """
        Check the data types of each column in the dataset and print them.
        """
        data_types = self.data.dtypes
        logger.info("Data types checked")
        print("\nData types of each column:")
        print(data_types)

    def analyze(self):
        """
        Perform analysis on the dataset by checking for missing values and data types.
        """
        logger.info("Analysis started")
        self.check_missing_values()
        self.check_data_types()
        print("Dataset shape: ", self.data.shape)
        print(self.data.head(10))
        logger.info("Analysis completed")
    
    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """
        initial_data_length = len(self.data)
        self.data = self.data.drop_duplicates()
        logger.info("Duplicates removed: %s records", initial_data_length - len(self.data))
        return self.data

    def get_column_names(self) -> list[str]:
        """
        Return the names of all columns in the dataset.
        """
        logger.info("Column names retrieved.")
        return self.data.columns.tolist()
