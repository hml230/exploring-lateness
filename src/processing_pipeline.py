"""This module performs as a data processing pipelien for 3 datasets sourced from OpenDataNSW"""
import warnings

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, DateType, StringType, StructField, IntegerType, DoubleType
import yaml
warnings.filterwarnings('ignore')


spark = SparkSession.builder.config("spark.driver.memory", "8g") \ 
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3").appName("Bus Performance Analysis").getOrCreate()


with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)


datasets = [
    ("2016_Occupancy_8-August-to-14-August.csv", "august"),
    ("2016_Occupancy_21-November-to-27-November.csv", "november"),
    ("2016_Occupancy_26-December-to-1-January-2017.csv", "eoy")
]


# Schema definition
schema_def = StructType([
    StructField("CALENDAR_DATE", DateType(), True),
    StructField("ROUTE", StringType(), True),
    StructField("ROUTE_VARIANT", StringType(), True),
    StructField("DIRECTION", StringType(), True),
    StructField("TIMETABLE_HOUR_BAND", StringType(), True),
    StructField("TRIP_CODE", StringType(), True),
    StructField("VEHICLETRIP", StringType(), True),
    StructField("TRIP_POINT", StringType(), True),
    StructField("TRANSIT_STOP", IntegerType(), True),
    StructField("TRANSIT_STOP_SEQUENCE", IntegerType(), True),
    StructField("TRANSIT_STOP_DESCRIPTION", StringType(), True),
    StructField("TIMETABLE_TIME", StringType(), True), # Cast to timestamp
    StructField("ACTUAL_TIME", StringType(), True), # Cast to timestamp
    StructField("SUBURB", StringType(), True),
    StructField("LATITUDE", DoubleType(), True),
    StructField("LONGITUDE", DoubleType(), True),
    StructField("DEPOT", StringType(), True),
    StructField("BUS_CONFIGURATION", StringType(), True),
    StructField("BUS_DOORS", IntegerType(), True),
    StructField("SEATED_CAPACITY", IntegerType(), True),
    StructField("STANDING_CAPACITY", IntegerType(), True),
    StructField("TOTAL_CAPACITY", IntegerType(), True),
    StructField("OPAL_RECORD_STATUS", StringType(), True),
    StructField("TIME_PERIOD", StringType(), True),
    StructField("CAPACITY_BUCKET", StringType(), True)
])


def process_single_dataset(file_name, dataset_label):
    """Process a single dataset with all cleaning and feature engineering steps"""    

    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("dateFormat", "dd/MMM/yy") \
        .schema(schema_def) \
        .load(config["raw_data_path"] + file_name)

    for c in df.columns:
        df = df.withColumnRenamed(c, c.lower())
    

    df = df.drop("trip_point", "opal_record_status", "timetable_hour_band",
                "latitude", "longitude",  "trip_code",
                "timetable_hour_band", "time_period", "transit_stop_description",
                "bus_doors", "vehicletrip")
    df = df.filter(df.actual_time.isNotNull())

    string_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
    for col_name in string_cols:
        df = df.withColumn(col_name, F.lower(F.trim(F.col(col_name))))
    
    df = df.withColumn(
        "actual_time",
        F.to_timestamp(
            F.concat(F.col("calendar_date"), F.lit(" "), F.col("actual_time")),
            "yyyy-MM-dd HH:mm"
        )
    )

    df = df.withColumn(
        "timetable_time",
        F.to_timestamp(
            F.concat(F.col("calendar_date"), F.lit(" "), F.col("timetable_time")),
            "yyyy-MM-dd HH:mm"
        )
    )

    # Create lateness variable
    df = df.withColumn(
        "lateness_minutes",
        ((F.unix_timestamp("actual_time") - F.unix_timestamp("timetable_time")) / 60).cast("double")
    )

    # Cap outlier values, lateness > 60 minutes suggests issues that are not inherent in buses
    df = df.withColumn(
    "lateness_minutes",
    F.when(F.col("lateness_minutes") < -60, -60)
     .when(F.col("lateness_minutes") > 60, 60)
     .otherwise(F.col("lateness_minutes"))
    )
    
    # Create lateness buckets
    df = df.withColumn(
        "lateness_bucket", 
        F.when(F.col("lateness_minutes") <= -1, "Early") 
         .when((F.col("lateness_minutes") > -1) & (F.col("lateness_minutes") <= 1), "On Time") 
         .when((F.col("lateness_minutes") > 1) & (F.col("lateness_minutes") <= 5), "1–5 min Late") 
         .when((F.col("lateness_minutes") > 5) & (F.col("lateness_minutes") <= 15), "5–15 min Late") 
         .when((F.col("lateness_minutes") > 15) & (F.col("lateness_minutes") <= 30), "15–30 min Late") 
         .when(F.col("lateness_minutes") > 30, "30+ min Late") 
         .otherwise("Unknown")
    )


    # Extract day of week (0 = Monday, 6 = Sunday)
    df = df.withColumn("day_of_week", F.dayofweek(F.col("calendar_date")) - 1)
    
    df = df.withColumn("timetable_hr", F.hour(F.col("timetable_time")))
    df = df.withColumn("timetable_min", F.minute(F.col("timetable_time")))
    
    df = df.withColumn("is_weekend", F.when(F.col("day_of_week").isin([5, 6]), 1).otherwise(0))
    
    # Encode cyclical features for hour (24-hour cycle)
    df = df.withColumn("timetable_hr_sin", 
                       F.sin(2 * F.lit(3.14159) * F.col("timetable_hr") / 24))
    df = df.withColumn("timetable_hr_cos", 
                       F.cos(2 * F.lit(3.14159) * F.col("timetable_hr") / 24))
    
    # Encode cyclical features for minute (60-minute cycle)
    df = df.withColumn("timetable_min_sin", 
                       F.sin(2 * F.lit(3.14159) * F.col("timetable_min") / 60))
    df = df.withColumn("timetable_min_cos", 
                       F.cos(2 * F.lit(3.14159) * F.col("timetable_min") / 60))
    
    # Encode cyclical features for day of week (7-day cycle)
    df = df.withColumn("day_of_week_sin", 
                       F.sin(2 * F.lit(3.14159) * F.col("day_of_week") / 7))
    df = df.withColumn("day_of_week_cos", 
                       F.cos(2 * F.lit(3.14159) * F.col("day_of_week") / 7))


    # Add label
    df = df.withColumn("dataset", F.lit(dataset_label))
    
    return df


def create_combined_dataset():
    """Process and combine all datasets"""
    
    joined_df = None
    
    for file_name, label in datasets:
        print(f"Processing {label} dataset...")
        df = process_single_dataset(file_name, label)
        
        if joined_df is None:
            joined_df = df
        else:
            joined_df = joined_df.unionAll(df)
    
    # Cache the final combined dataset
    joined_df.persist(StorageLevel.DISK_ONLY)
    
    return joined_df

    
def prepare_ml_features(df):
    """Prepare features specifically for ML models"""
    
    ml_features = [
        "calendar_date", "timetable_time", "route", "route_variant", "direction",
        "transit_stop", "transit_stop_sequence", "suburb", "depot", "bus_configuration",
        "seated_capacity", "standing_capacity", "total_capacity", "capacity_bucket", "dataset",
        "is_weekend", "timetable_hr_sin", "timetable_hr_cos", "timetable_min_sin", "timetable_min_cos",
        "day_of_week_sin", "day_of_week_cos", "lateness_minutes", "lateness_bucket"
    ]
    
    return df.select(*ml_features)


def export_stratified_samples(df, sample_sizes=[10000, 50000, 100000]):
    """Export stratified samples for training"""
    
    for size in sample_sizes:
        # Calculate fraction needed for each bucket
        bucket_counts = df.groupBy("lateness_bucket").count().collect()
        total_count = df.count()
        
        # Create stratified sample
        sampled_dfs = []
        for row in bucket_counts:
            bucket = row["lateness_bucket"]
            bucket_count = row["count"]
            fraction = min(1.0, (size * bucket_count / total_count) / bucket_count)
            
            bucket_sample = df.filter(F.col("lateness_bucket") == bucket).sample(fraction, seed=42)
            sampled_dfs.append(bucket_sample)
        
        # Union all bucket samples
        stratified_sample = sampled_dfs[0]
        for i in range(1, len(sampled_dfs)):
            stratified_sample = stratified_sample.unionAll(sampled_dfs[i])
        
        # Export to CSV
        output_path = f"{config['processed_path']}sample_{size}.csv"
        stratified_sample.repartition(1).write.options(header=True).csv(output_path, mode="overwrite")
        print(f"Exported stratified {size} samples to {output_path}")

def main():
    """Main processing pipeline"""
    
    combined_df = create_combined_dataset()
    
    ml_ready_df = prepare_ml_features(combined_df)
    
    full_output_path = f"{config['processed_path']}combined_processed_data.csv"
    ml_ready_df.repartition(2).write.options(header=True).csv(full_output_path, mode="overwrite")
    print(f"Exported full processed dataset to {full_output_path}")
    
    export_stratified_samples(ml_ready_df)
    
    print("\n=== DATASET SUMMARY ===")
    print(f"Total records: {ml_ready_df.count():,}")
    print("\nLateness distribution:")
    ml_ready_df.groupBy("lateness_bucket").count().orderBy("count", ascending=False).show()
    
    print("\nDataset distribution:")
    ml_ready_df.groupBy("dataset").count().show()
    
    # Unpersist cache
    combined_df.unpersist()

if __name__ == "__main__":
    main()
    spark.catalog.clearCache()
    spark.stop()
    
