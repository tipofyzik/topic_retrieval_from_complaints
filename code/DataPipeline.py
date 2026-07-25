from DataAnalyzer import DataAnalyzer
from SensorMongoDB import SensorMongoDB

import pandas as pd
import os

class DataPipeline:
    """
    A class to handle the data pipeline for IoT telemetry data.
    """
    def __init__(self):
        """
        Initialize the DataPipeline with a pandas DataFrame.
        
        Params: 
            data (pd.DataFrame): A pandas DataFrame containing the IoT telemetry data.
        """
        self.iot_data = pd.read_csv("./data/iot_telemetry_data.csv")
        self.__convert_timestamps_to_date()
        self.available_dates = (
            self.iot_data["ts"]
            .dt.floor("D")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        self.batch_size = 1000
        self.base_columns = ['ts', 'device', 'temp', 'humidity']
        self.extra_columns = ['ts', 'device','co', 'lpg', 'smoke', 'motion', 'light']

        self.n_records = 5
        self.update_rows = None
        self.test_rows = None
        self.historical_analysis_log = "./logs/analysis_history_log.csv"
        self.update_history_log = "./logs/update_history_logs.csv"

    def __convert_timestamps_to_date(self):
        """
        Convert Unix timestamps into readable datetime values.
        Keeps original ts column for database operations.
        """
        self.iot_data["ts"] = pd.to_datetime(
            self.iot_data["ts"],
            unit="s"
        )

    def analyze_and_clean_data(self, strategy: str = "drop"):
        """
        Analyze and clean the IoT telemetry data.
        """
        data_analyzer = DataAnalyzer(self.iot_data)
        data_analyzer.analyze()
        data_analyzer.handle_missing_values(strategy=strategy)
        self.iot_data = data_analyzer.remove_duplicates()

    def connect_to_database(self):
        """
        Connect to the MongoDB database and check the connection.
        """
        self.database = SensorMongoDB()
        if not self.database.check_connection():
            raise ConnectionError("Failed to connect to MongoDB.")

    def clear_database(self):
        """
        Clear the MongoDB database.
        It should be cleaned as MongoDB stores 
        """
        self.database.clear()
        self.update_rows = None
        self.test_rows = None

    def insert_data_in_batches(self):
        """
        Insert the base columns of the IoT telemetry data into MongoDB in batches.
        """
        self.database.insert_batch(data_list=self.iot_data[self.base_columns].to_dict("records"), 
                                   batch_size=self.batch_size)
        print("Inserted all data into MongoDB in batches of {}.".format(self.batch_size))
        print("\nTotal rows in the dataset: {}".format(len(self.iot_data)))
        print(f"Total documents in collection: {self.database.count_documents()}\n")

    def prepare_update_data(
                self,
                date: str,
                hour: str,
                duration: str
        ):
        """
        Prepare records for update based on selected time interval.
        """
        durations = {
            "1 Hour": pd.Timedelta(hours=1),
            "6 Hours": pd.Timedelta(hours=6),
            "12 Hours": pd.Timedelta(hours=12),
            "1 Day": pd.Timedelta(days=1),
            "3 Days": pd.Timedelta(days=3),
            "7 Days": pd.Timedelta(days=7),
            "30 Days": pd.Timedelta(days=30)
        }

        self.start = pd.Timestamp(
            f"{date} {hour}:00:00"
        )
        self.end = self.start + durations[duration]

        self.update_rows = self.iot_data[
            (self.iot_data["ts"] >= self.start) &
            (self.iot_data["ts"] <= self.end)
        ]

        if self.update_rows.empty:
            print("No records found for update.")
            return
        
        self.test_rows = self.update_rows.sample(
            min(self.n_records, len(self.update_rows))
        )

        print("\n===== Update preparation =====")
        print(f"From    : {self.start}")
        print(f"To      : {self.end}")
        print(f"Records : {len(self.update_rows)}")

    def update_existing_records(self):
        if self.update_rows is None or self.update_rows.empty:
            print("No update records available.")
            return

        self.database.update_batch(
            data_list=self.update_rows[self.extra_columns].to_dict("records"),
            batch_size=self.batch_size
        )
        self.save_update_result(
            strategy="update_existing_records",
            start=self.start,
            end=self.end,
            records=len(self.update_rows),
            columns=[
                col for col in self.extra_columns
                if col not in ["ts", "device"]
            ]
        )

    def save_update_result(
            self,
            strategy,
            start,
            end,
            records,
            columns
    ):
        result = pd.DataFrame(
            [
                {
                    "strategy": strategy,
                    "from": start,
                    "to": end,
                    "records": records,
                    "updated_columns": ", ".join(columns)
                }
            ]
        )

        file_exists = (
            os.path.exists(self.update_history_log)
            and os.path.getsize(self.update_history_log) > 0
        )

        result.to_csv(
            self.update_history_log,
            mode="a",
            header=not file_exists,
            index=False
        )

    def alternative_update_strategy(self):
        """
        An alternative strategy to update existing records in MongoDB by adding new documents.
        This preserves timestamps and device identifiers, which helps in matching new data with old data.
        """
        if self.update_rows is None:
            print("No update records available.")
            return
        self.database.insert_batch(data_list=self.update_rows[self.extra_columns].to_dict("records"), 
                                   batch_size=self.batch_size)
        self.database.check_connection()
        # self.check_latest_records()

    def get_available_dates(self):
        return self.available_dates

    def get_available_hours(self, selected_date):
        """
        Return available hours for selected date.
        """

        selected_date = pd.to_datetime(selected_date)

        hours = (
            self.iot_data[
                self.iot_data["ts"].dt.date == selected_date.date()
            ]["ts"]
            .dt.hour
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return hours

    def analyze_history(
            self,
            feature: str,
            date: str,
            hour: str,
            duration: str
    ):

        durations = {
            "1 Hour": pd.Timedelta(hours=1),
            "6 Hours": pd.Timedelta(hours=6),
            "12 Hours": pd.Timedelta(hours=12),
            "1 Day": pd.Timedelta(days=1),
            "3 Days": pd.Timedelta(days=3),
            "7 Days": pd.Timedelta(days=7),
            "30 Days": pd.Timedelta(days=30)
        }


        start = pd.Timestamp(
            f"{date} {hour}:00:00"
        )

        end = start + durations[duration]


        database_data = self.load_data_from_database()


        if database_data is None:
            return


        history = database_data[
            (database_data["ts"] >= start) &
            (database_data["ts"] <= end)
        ]


        if history.empty:
            print("No records found.")
            return


        stats = history[feature].agg(
            [
                "min",
                "max",
                "mean"
            ]
        )

        self.save_analysis_result(
            feature=feature,
            start=start,
            end=end,
            records=len(history),
            stats=stats
        )

        print("\n===== Historical analysis =====")
        print(f"Sensor  : {feature}")
        print(f"From    : {start}")
        print(f"To      : {end}")
        print(f"Records : {len(history)}")
        print()
        print(stats)

    def save_analysis_result(
            self,
            feature,
            start,
            end,
            records,
            stats
    ):
        """
        Save historical analysis results to CSV log.
        """
        result = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "from": start,
                    "to": end,
                    "records": records,
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"]
                }
            ]
        )

        file_exists = (
            os.path.exists(self.historical_analysis_log)
            and os.path.getsize(self.historical_analysis_log) > 0
        )
        result.to_csv(
            self.historical_analysis_log,
            mode="a",
            header=not file_exists,
            index=False
        )

    def show_analysis_history(self):

        try:
            history = pd.read_csv(
                self.historical_analysis_log
            )

            if history.empty:
                print("No analyses performed.")
                return

            print("\n===== Analysis history =====")
            print(history)

        except FileNotFoundError:
            print("No analysis history available.")

    def show_update_history(self):

        try:
            history = pd.read_csv(
                self.update_history_log
            )

            if history.empty:
                print("No update history available.")
                return

            print("\n===== Update history =====")
            print(history)

        except FileNotFoundError:
            print("No update history available.")

    def clear_analysis_history(self):
        open(self.historical_analysis_log, "w").close()
        print("Analysis history cleared.")

    def clear_update_history(self):
        open(self.update_history_log, "w").close()
        print("Analysis history cleared.")

    def retrieve_by_unique_value(self, key: str):
        values = self.database.get_unique_values(key)
        if not values:
            print(f"No data available for field '{key}'.")
            return
        print(f"\nUnique values for {key}:")
        for i, value in enumerate(values):
            print(f"{i+1}. {value}")
        selected = int(input("Select value number: ")) - 1
        selected_value = values[selected]
        records = self.database.get_by_field(
            key,
            selected_value
        )
        print(f"\nFound {len(records)} records")
        for record in records[:5]:
            print(record)

    def load_data_from_database(self):
        """
        Load current sensor data from MongoDB
        for analysis.
        """

        records = self.database.get_all_records()

        if not records:
            print("Database is empty.")
            return None


        data = pd.DataFrame(records)

        data["ts"] = pd.to_datetime(
            data["ts"]
        )


        return data