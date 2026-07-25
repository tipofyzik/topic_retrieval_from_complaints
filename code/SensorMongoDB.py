from pymongo import MongoClient, UpdateOne
import logging

logging.basicConfig(
    filename="./logs/database.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)



class SensorMongoDB:
    """
    Simple wrapper class for MongoDB sensor data storage.
    Designed for IoT batch ingestion.
    """

    def __init__(self, uri="mongodb://root:example@mongodb:27017/", db_name="sensordb"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["sensor_readings"]
        self.collection.create_index(
            [
                ("ts", 1),
                ("device", 1)
            ],
            name="timestamp_device_unique"
        )
        logger.info(
            "MongoDB client initialized. Database: %s, Collection: sensor_readings",
            db_name
        )

    def check_connection(self):
        try:
            self.client.admin.command("ping")
            logger.info(
                "MongoDB connection successful"
            )

            print("MongoDB connection: OK")
            return True
        except Exception as e:
            logger.exception(
                "MongoDB unavailable: %s",
                e
            )
            print("MongoDB connection failed:", e)
            return False

    # -----------------------------
    # Single insert (for testing)
    # -----------------------------
    def insert_reading(self, data: dict):
        """
        Insert single sensor reading.
        Automatically adds timestamp if missing.

        Params:
            data (dict): A dictionary containing sensor reading data.        
        """
        try:
            result = self.collection.insert_one(data)
            logger.info(
                "Inserted single sensor reading. ID: %s",
                result.inserted_id
            )
            print("Sensor reading inserted successfully")
            return result
        except Exception as e:
            logger.exception(
                "Failed to insert sensor reading: %s",
                e
)
            print("Failed to insert sensor reading:", e)
            raise e

    # -----------------------------
    # Batch insert (main requirement)
    # -----------------------------
    def insert_batch(self, data_list: list[dict], batch_size: int = 1000):
        """
        Insert sensor data in batches.

        Params:
            data_list (list[dict]): A list of dictionaries containing sensor reading data.
            batch_size (int): Number of records to insert in each batch. Default is 1000.
        """
        logger.info("Starting batch insertion. Total records: %s, Batch size: %s", len(data_list), batch_size)

        try:            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                self.collection.insert_many(
                    batch,
                    ordered=False
                )
                logger.info("Inserted batch %s-%s", i, i + len(batch))
            logger.info("Batch insertion completed. Total inserted records: %s", len(data_list))
        except Exception as e:
            logger.exception("Batch insertion failed: %s", e)
            print("Batch insertion failed:", e)
            raise

    def update_batch(
            self,
            data_list: list[dict],
            batch_size: int = 1000
    ):
        logger.info(
            "Starting batch update. Records: %s, Batch size: %s",
            len(data_list),
            batch_size
        )

        try:
            operations = []
            total_matched = 0
            total_modified = 0

            for item in data_list:
                update_fields = {
                    key: value
                    for key, value in item.items()
                    if key not in ["ts", "device"]
                }

                operations.append(
                    UpdateOne(
                        {
                            "ts": item["ts"],
                            "device": item["device"]
                        },
                        {
                            "$set": update_fields
                        }
                    )
                )

                if len(operations) == batch_size:
                    result = self.collection.bulk_write(
                        operations,
                        ordered=False
                    )

                    total_matched += result.matched_count
                    total_modified += result.modified_count
                    operations = []

            if operations:
                result = self.collection.bulk_write(
                    operations,
                    ordered=False
                )

                total_matched += result.matched_count
                total_modified += result.modified_count

            logger.info(
                "Update finished. Matched: %s Modified: %s",
                total_matched,
                total_modified
            )

            print(f"Matched: {total_matched}")
            print(f"Modified: {total_modified}")

        except Exception as e:
            logger.exception(
                "Batch update failed: %s",
                e
            )
            raise

    # -----------------------------
    # Query readyings by filter (for testing)
    # -----------------------------
    def get_by_keys(self, keys: list[tuple]):
        """
        keys: [(ts, device), ...]
        """
        try:
            records = list(
                self.collection.find({
                    "$or": [
                        {"ts": ts, "device": device}
                        for ts, device in keys
                    ]
                })
            )
            logger.info("Retrieved %s records by keys", len(records))
            return records
        except Exception as e:
            logger.exception("Failed to retrieve records by keys: %s", e)
            print("Failed to retrieve records by keys:", e)
            raise

    def get_unique_values(self, key: str):
        """
        Retrieve unique values for a specific field.
        """
        try:
            values = self.collection.distinct(key)
            if not values:
                logger.warning("No unique values found for key: %s", key)
                print(f"No values found for field '{key}'.")
                return 
            logger.info("Retrieved %s unique values for key: %s", len(values), key)
            return values
        except Exception as e:
            logger.exception("Failed to retrieve unique values: %s", e)
            print("Failed to retrieve unique values:", e)
            raise

    def get_by_field(self, key: str, value):
        """
        Retrieve documents by specific field value.
        """
        try:
            records = list(
                self.collection.find({
                    key: value
                })
            )
            logger.info(
                "Retrieved %s records where %s=%s",
                len(records),
                key,
                value
            )
            return records
        except Exception as e:
            logger.exception("Failed to retrieve records by field: %s", e)
            print("Failed to retrieve records:", e)
            raise

    def count_documents(self):
        try:
            count = self.collection.count_documents({})
            logger.info("Total documents in collection: %s", count)
            return count
        except Exception as e:
            logger.exception("Failed to count documents: %s", e)
            print("Failed to count documents", e)
            raise

    def clear(self):
        """
        Delete all data. Required as MongoDB stores documents as seperate objects. 
        If not cleared, the database will store duplicate of the entire dataset. 
        """
        try:
            result = self.collection.delete_many({})
            logger.info("Database cleared. Deleted documents: %s", result.deleted_count)
            print(f"Database cleared. Deleted documents: {result.deleted_count}")
        except Exception as e:
            logger.exception("Failed to clear database: %s", e)
            print("Failed to clear database:", e)
            raise

    def get_all_records(self):
        """
        Retrieve all sensor records from MongoDB.
        """
        try:
            records = list(
                self.collection.find(
                    {},
                    {"_id": 0}
                )
            )

            logger.info(
                "Retrieved %s records from database",
                len(records)
            )

            return records

        except Exception as e:
            logger.exception(
                "Failed to retrieve records: %s",
                e
            )
            raise