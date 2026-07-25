// Connect to the 'sensordb' database
db = db.getSiblingDB('sensordb');

// Creating time-series collection 'sensor_readings' for storing sensor data
// 'timeField' - mandatory field with timestamp
// 'metaField' - optional field to identify the device generating the data
db.createCollection(
    "sensor_readings",
    {
        timeseries: {
            timeField: "ts",
            metaField: "device"
        },
        expireAfterSeconds: 2592000 // Optional: TTL of 30 days
    }
);

// Creating additional index for faster time-based queries
db.sensor_readings.createIndex({ "ts": -1 });