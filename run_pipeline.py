from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "diabetic_data.csv"

if not DATA_FILE.exists():
    print("\nDATASET NOT FOUND.")
    print("Download the UCI Diabetes 130-US Hospitals dataset:")
    print("https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008")
    print(f"Copy diabetic_data.csv into: {DATA_FILE}")
    sys.exit(1)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, trim, when, count, avg, xxhash64, pmod, lit
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
except ImportError as exc:
    print("A Python package is missing. Run setup_windows.bat first.")
    raise exc

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

spark = (
    SparkSession.builder
    .appName("HealthcarePredictiveAnalytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n[1/5] Reading the CSV with Apache Spark...")
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(str(DATA_FILE))
)
initial_count = df.count()
print(f"Rows loaded: {initial_count:,}")
print(f"Columns: {len(df.columns)}")

required = {"readmitted", "patient_nbr", "age", "gender"}
missing = required - set(df.columns)
if missing:
    spark.stop()
    raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

print("\n[2/5] Cleaning data...")
for name in df.columns:
    if df.schema[name].dataType.simpleString() == "string":
        df = df.withColumn(
            name,
            when(trim(col(name)) == "?", None).otherwise(trim(col(name)))
        )

df = df.dropDuplicates()
# The target has three known categories in this dataset. Ignore any unexpected/missing target.
df = df.filter(col("readmitted").isin("<30", ">30", "NO"))
# A missing patient identifier cannot be used for group-aware splitting.
df = df.filter(col("patient_nbr").isNotNull())

df = df.withColumn("label", (col("readmitted") == "<30").cast("double"))

numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]
categorical_cols = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "change", "diabetesMed"
]

for name in numeric_cols:
    if name in df.columns:
        df = df.withColumn(name, col(name).cast("double"))

# Ensure only features present in the CSV are used.
numeric_cols = [c for c in numeric_cols if c in df.columns]
categorical_cols = [c for c in categorical_cols if c in df.columns]

cleaned_count = df.count()
print(f"Rows after cleaning: {cleaned_count:,}")

print("\n[3/5] Running Spark SQL analytics...")
df.createOrReplaceTempView("healthcare_encounters")
spark.sql("""
    SELECT readmitted, COUNT(*) AS encounters
    FROM healthcare_encounters
    GROUP BY readmitted
    ORDER BY encounters DESC
""").show(truncate=False)

age_summary = spark.sql("""
    SELECT age,
           COUNT(*) AS encounters,
           AVG(label) AS early_readmission_rate
    FROM healthcare_encounters
    GROUP BY age
    ORDER BY age
""")
age_summary.toPandas().to_csv(ARTIFACTS / "analytics_summary.csv", index=False)

processed_path = ARTIFACTS / "processed_healthcare"
if os.name == "nt":
    # Spark's local Hadoop writer requires winutils.exe on Windows, even without HDFS.
    processed_csv_path = ARTIFACTS / "processed_healthcare.csv"
    df.toPandas().to_csv(processed_csv_path, index=False)
    print(
        "Windows detected: saved cleaned data as CSV because Spark Parquet output "
        f"requires winutils.exe. Saved to: {processed_csv_path}"
    )
else:
    df.write.mode("overwrite").parquet(str(processed_path))
    print(f"Cleaned Parquet data saved to: {processed_path}")

print("\n[4/5] Training Spark MLlib Logistic Regression...")
# Split by patient number to reduce leakage from multiple encounters for the same patient.
df = df.withColumn("_split_bucket", pmod(xxhash64(col("patient_nbr")), lit(5)))
train = df.filter(col("_split_bucket") != 0).drop("_split_bucket")
test = df.filter(col("_split_bucket") == 0).drop("_split_bucket")

train_count, test_count = train.count(), test.count()
print(f"Training rows: {train_count:,}; test rows: {test_count:,}")

if train_count == 0 or test_count == 0:
    spark.stop()
    raise ValueError("Training/test split is empty. Check the dataset.")

for name in categorical_cols:
    df = df.withColumn(name, when(col(name).isNull(), "Unknown").otherwise(col(name)))
    train = train.withColumn(name, when(col(name).isNull(), "Unknown").otherwise(col(name)))
    test = test.withColumn(name, when(col(name).isNull(), "Unknown").otherwise(col(name)))

imputed_cols = [f"{c}_imputed" for c in numeric_cols]
stages = []
for c in categorical_cols:
    stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))

if categorical_cols:
    stages.append(OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_cols],
        outputCols=[f"{c}_vec" for c in categorical_cols],
        handleInvalid="keep"
    ))

if numeric_cols:
    stages.append(Imputer(
        inputCols=numeric_cols,
        outputCols=imputed_cols,
        strategy="median"
    ))

feature_cols = imputed_cols + [f"{c}_vec" for c in categorical_cols]
stages.append(VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep"))
stages.append(LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    regParam=0.05
))

pipeline = Pipeline(stages=stages)
model = pipeline.fit(train)
predictions = model.transform(test).cache()

accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)
precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
).evaluate(predictions)
recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
).evaluate(predictions)
f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
).evaluate(predictions)

try:
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)
except Exception:
    auc = None

metrics = {
    "initial_records": int(initial_count),
    "cleaned_records": int(cleaned_count),
    "training_records": int(train_count),
    "testing_records": int(test_count),
    "accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc) if auc is not None else None,
    "target_definition": "readmitted == <30"
}
import json
with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# A small prediction sample for the dashboard; no patient identifier is exported.
from pyspark.ml.functions import vector_to_array
sample = predictions.select(
    "age", "gender", "time_in_hospital", "label", "prediction",
    vector_to_array("probability")[1].alias("readmission_probability")
).limit(2000)
sample.toPandas().to_csv(ARTIFACTS / "predictions_sample.csv", index=False)

model_path = ARTIFACTS / "readmission_model"
if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
    print(
        "Windows detected: skipped Spark model export because it requires "
        "winutils.exe. Set HADOOP_HOME to a Hadoop helper directory and rerun "
        "the pipeline to enable manual predictions."
    )
else:
    model.write().overwrite().save(str(model_path))

print("\n[5/5] Results")
print(json.dumps(metrics, indent=2))
print(f"\nArtifacts saved in: {ARTIFACTS}")
print("Next: run run_dashboard.bat or 'streamlit run app.py'.")

predictions.unpersist()
spark.stop()
