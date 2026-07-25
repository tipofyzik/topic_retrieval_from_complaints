from DataPipeline import DataPipeline
import questionary as q



class CommandLineInterface:

    def __init__(self):
        self.pipeline = DataPipeline()

    def start_menu(self):
        choice_handler = {
            "Analyze and clean initial data": self.analyze_data,
            "Connect to database": self.connect_database,
            "Clear database": self.clear_database,
            "Data manipulation": self.data_manipulation_menu,
            "Database analysis": self.analysis_menu,
            "View logs": self.view_logs,
            "Quit": self.quit_program
        }

        question = q.select(
            "What do you want to do?",
            choices=[
                "Analyze and clean initial data",
                "Connect to database",
                "Clear database",
                "Data manipulation",
                "Database analysis",
                "View logs",
                "Quit"
            ]
        ).ask()
        choice_handler[question]()

    def data_manipulation_menu(self):
        choice_handler = {
            "Insert sensor data": self.insert_data,
            "Update existing records": self.update_records,
            "Alternative update strategy": self.alternative_update,
            "Return back": self.start_menu
        }

        question = q.select(
            "Select data manipulation:",
            choices=[
                "Insert sensor data",
                "Update existing records",
                "Alternative update strategy",
                "Return back"
            ]
        ).ask()
        choice_handler[question]()

    def analysis_menu(self):
        choice_handler = {
            "Database information": self.database_information,
            "Historical analysis": self.historical_analysis,
            "Retrieve records": self.retrieve_records,
            "Return back": self.start_menu
        }

        question = q.select(
            "Select analysis option:",
            choices=[
                "Database information",
                "Historical analysis",
                "Retrieve records",
                "Return back"
            ]
        ).ask()
        choice_handler[question]()

    def analyze_data(self):
        strategy = q.select(
            "Select missing values handling strategy:",
            choices=[
                "drop",
                "fill"
            ]
        ).ask()
        self.pipeline.analyze_and_clean_data(strategy = strategy)
        print("Data was analyzed, cleaned and duplicates were removed.")

    def connect_database(self):
        try:
            self.pipeline.connect_to_database()
            print("MongoDB connection established.")
        except ConnectionError as error:
            print(error)

    def clear_database(self):
        choice_handler = {
            "Yes": self.pipeline.clear_database,
            "No": self.start_menu
        }

        question = q.select(
            "Are you sure you want to clear database?",
            choices=[
                "Yes",
                "No"
            ]
        ).ask()

        choice_handler[question]()
        if question == "Yes":
            print("Database successfully cleared.")



    def insert_data(self):
        self.pipeline.insert_data_in_batches()

    def update_records(self):
        dates = self.pipeline.get_available_dates()
        selected_date = q.select(
            "Select update date:",
            choices=[
                str(date)
                for date in dates
            ]
        ).ask()


        hours = self.pipeline.get_available_hours(
            selected_date
        )
        selected_hour = q.select(
            "Select update hour:",
            choices=[
                str(hour)
                for hour in hours
            ]
        ).ask()

        duration = q.select(
            "Select update period:",
            choices=[
                "1 Hour",
                "6 Hours",
                "12 Hours",
                "1 Day",
                "3 Days",
                "7 Days",
                "30 Days"
            ]
        ).ask()

        self.pipeline.prepare_update_data(
            selected_date,
            selected_hour,
            duration
        )
        self.pipeline.update_existing_records()

    def alternative_update(self):
        self.pipeline.alternative_update_strategy()

    def show_document_count(self):
        count = self.pipeline.database.count_documents()
        print(
            f"Total documents in collection: {count}"
        )

    def database_information(self):
        info_handler = {
            "Count documents": self.show_document_count,
            "Return back": self.start_menu
        }

        question = q.select(
            "Select information:",
            choices=[
                "Count documents",
                "Return back"
            ]
        ).ask()

        info_handler[question]()



    def historical_analysis(self):
        feature = q.select(
            "Select sensor:",
            choices=[
                "temp",
                "humidity",
                "co",
                "lpg",
                "smoke",
                "light"
            ]
        ).ask()

        dates = self.pipeline.get_available_dates()
        selected_date = q.select(
            "Select date:",
            choices=[
                str(date)
                for date in dates
            ]
        ).ask()

        hours = self.pipeline.get_available_hours(selected_date)
        selected_hour = q.select(
            "Select hour:",
            choices=[
                str(hour)
                for hour in hours
            ]
        ).ask()

        duration = q.select(
            "Select analysis period:",
            choices=[
                "1 Hour",
                "6 Hours",
                "12 Hours",
                "1 Day",
                "3 Days",
                "7 Days",
                "30 Days"
            ]
        ).ask()

        self.pipeline.analyze_history(
            feature,
            selected_date,
            selected_hour,
            duration
        )

    def retrieve_records(self):
        fields = ['ts', 'device', 'temp', 'humidity']
        key = q.select(
            "Select field:",
            choices=fields
        ).ask()
        self.pipeline.retrieve_by_unique_value(key)

    def view_logs(self):
        log_handler = {
            "Database logs": self.show_database_logs,
            "Analysis history": self.pipeline.show_analysis_history,
            "Update history": self.pipeline.show_update_history,
            "Clear database logs": self.clear_database_logs,
            "Clear analysis history": self.pipeline.clear_analysis_history,
            "Clear update history": self.pipeline.clear_update_history,
            "Return back": self.start_menu
        }

        question = q.select(
            "Select log type:",
            choices=[
                "Database logs",
                "Analysis history",
                "Update history",
                "Clear database logs",
                "Clear analysis history",
                "Clear update history",
                "Return back"
            ]
        ).ask()

        log_handler[question]()

    def show_database_logs(self):
        log_file = "./logs/database.log"
        try:
            with open(log_file, "r") as file:
                logs = file.readlines()
            if not logs:
                print("Log file is empty.")
                return
            print("\n===== DATABASE LOGS =====")
            for line in logs[-20:]:
                print(line.strip())
        except FileNotFoundError:
            print("No log file found.")

    def clear_database_logs(self):
        open("./logs/database.log", "w").close()
        print("Database logs cleared.")

    def quit_program(self):
        choice_handler = {
            "Yes": exit,
            "No": self.start_menu
        }
        question = q.select(
            "Are you sure you want to quit?",
            choices=[
                "Yes",
                "No"
            ]
        ).ask()
        choice_handler[question]()
