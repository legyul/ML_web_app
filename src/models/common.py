import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as sklearnStandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import re
from dotenv import load_dotenv
import os
from logger_utils import logger
import boto3
from botocore.exceptions import ClientError
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf
from pyspark.ml.feature import Imputer, StringIndexer, StandardScaler as sparkStandardScaler, VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType
from pyspark.sql import functions as F


load_dotenv()
os.environ["JAVA_HOME"] = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ["SPARK_HOME"] = '/usr/local/spark'

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

S3_BUCKET_NAME = "ml-platform-service"

s3 = boto3.client('s3')

# Load dataset file
def load_file(file_key):
    if not file_key:
        raise ValueError("Error: file_key is None. Check the function call.")
    
    file_name = file_key.split('/')[-1]
    file_path = f"uploaded/{file_name}"
    s3_path = f"s3://{S3_BUCKET_NAME}/{file_path}"

    if not file_name or not file_path:
        raise ValueError("Error: file_path is not generated correctly.")
    
    # Check if the file exists in S3 bucket
    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
        file_size = response['ContentLength']
        print(f"File exists in S3: {file_path}, size: {file_size} bytes")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise FileNotFoundError(f"File '{file_name}' does not exist in S3 bucket '{S3_BUCKET_NAME}'")
        else:
            raise e
    
    # Determine file extension
    file_extension = file_name.split('.')[-1]

    max_file_size = 100 * 1024 * 1024 # 100MB

    if file_size > max_file_size:
        mode = "spark"
    
        # Read the file based on its extension with PySpark
        if file_extension == 'csv':
            return spark.read.csv(s3_path, header=True, inferSchema=True), mode
        elif file_extension == 'json':
            return spark.read.json(s3_path), mode
        else:
            raise ValueError("Unsupported file format. Supported formats are .csv, .xlsx, and .json")
    
    else:
        mode = "pandas"

        # Read the file based on its extension with Pandas
        if file_extension == 'csv':
            return pd.read_csv(s3_path), mode
        elif file_extension == 'xlsx':
            return pd.read_excel(s3_path), mode
        elif file_extension == 'json':
            return pd.read_json(s3_path), mode
        else:
            raise ValueError("Unsupported file format. Supported formats are .csv, .xlsx, and .json")

class spark_processing:
    def spark_preprocessing_data(data, mode):
        if mode != "spark":
            data, gender_mapping = pandas_processing.pandas_preprocessing_data(data, mode)
            return data, gender_mapping

        data = data.na.drop(how='all')

        for col_name in data.columns:
            try:
                data = data.withColumn(col_name, col(col_name).cast(DoubleType()))
            except Exception as e:
                print(f"Error converting column {col_name}: {e}")

        numeric_cols = [field.name for field in data.schema.fields if isinstance(field.dataType, (DoubleType, FloatType, IntegerType, LongType))]
        imputer = Imputer(strategy="mean", inputCols=numeric_cols, outputCols=[f"{c}_imputed" for c in numeric_cols])

        try:
            df_imputed = imputer.fit(data).transform(data)
            imputed_columns = []
            for c in numeric_cols:
                imputed_col = f"{c}_imputed"
                if imputed_col in df_imputed.columns:
                    imputed_columns.append(F.col(imputed_col).alias(c))
                else:
                    imputed_columns.append(F.col(c))

            non_numeric_cols = [F.col(c) for c in data.columns if c not in numeric_cols]

            if len(imputed_columns) + len(non_numeric_cols) == len(data.columns):
                data = df_imputed.select(imputed_columns + non_numeric_cols)
            else:
                print("Column mismatch: imputed columns and non numeric cols lengths do not match.")
            
        except Exception as e:
            print("Error during imputation:", e)

        gender_mapping = {}
        
        if 'sex' in data.columns:
            data, gender_mapping = spark_processing.spark_process_gender_column(data, 'sex')
        elif 'gender' in data.columns:
            data, gender_mapping = spark_processing.spark_process_gender_column(data, 'gender')

        return data, gender_mapping

    def spark_standardize_gender(value):
        if value is None:
            return "unknown"
        
        value = value.lower()
        if "male" in value or "man" in value or "boy" in value:
            return "male"
        elif "female" in value or "woman" in value or "girl" in value:
            return "female"
        else:
            return "unknown"

    standardize_gender_udf = udf(spark_standardize_gender, StringType())

    def spark_process_gender_column(data, column_name):
        data = data.withColumn(column_name, spark_processing.standardize_gender_udf(lower(col(column_name))))

        indexer = StringIndexer(inputCol=column_name, outputCol=f'{column_name}')
        try:
            model = indexer.fit(data)
            gender_mapping = {value: index for index, value in enumerate(model.labels)}
            return data, gender_mapping
        except Exception as e:
            print(f"Error processing '{column_name}' column: {e}")
            return data, {}
        
    def spark_scale_df(data):
        numeric_cols = [field.name for field in data.schema.fields if isinstance(field.dataType, (DoubleType, FloatType, IntegerType, LongType))]

        for col_name in numeric_cols:
            # Assemble the column into a vector
            assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vector")
            data = assembler.transform(data)

            # Scale the vectorized column
            scaler = sparkStandardScaler(inputCol=f"{col_name}_vector", outputCol=f"{col_name}_scaled_vector")
            scaler_model = scaler.fit(data)
            data = scaler_model.transform(data)

            # Extract the scaled value from the vector and replace the original column
            data = data.withColumn(col_name, F.col(f"{col_name}_scaled_vector")[0])
            
            # Drop intermediate vector columns
            data = data.drop(f"{col_name}_vector", f"{col_name}_scaled_vector")

        return data

class pandas_processing:
    def pandas_preprocessing_data(data, mode):
        if mode != "pandas":
            data, gender_mapping = spark_processing.spark_preprocessing_data(data, mode)
            return data, gender_mapping

        data = data.copy()
        data = data.dropna(how="all")

        for col_name in data.columns:
            if data[col_name].dtype == 'object':
                print(f"Skipping non-numeric column: {col_name}")
            
            else:
                try:
                    data[col_name] = pd.to_numeric(data[col_name], errors='coerce')
                
                except Exception as e:
                    print(f"Error convering column {col_name}: {e}")
        
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy="mean")
        try:
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

        except Exception as e:
            print("error during imputation: ", e)

        gender_mapping = {}

        if 'sex' in data.columns:
            data, gender_mapping = pandas_processing.pandas_process_gender_column(data, 'sex')
        
        elif 'gender' in data.columns:
            data, gender_mapping = pandas_processing.pandas_process_gender_column(data, 'gender')
        
        return data, gender_mapping

    def pandas_standardize_gender(value):
        value = str(value).lower()
        
        if value in ["female", "woman", "girl"]:
            return "female"
        elif value in ["male", "man", "boy"]:
            return "male"
        else:
            return "unknown"

    def pandas_process_gender_column(data, column_name):

        data[column_name] = data[column_name].apply(pandas_processing.pandas_standardize_gender)

        label_encoder = LabelEncoder()

        try:
            data[f"{column_name}"] = label_encoder.fit_transform(data[column_name])
            gender_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
            print(gender_mapping)
            return data, gender_mapping
        
        except Exception as e:
            print(f"Error processing '{column_name}' column: {e}")
            return data, {}
    
    def pandas_scale_df(data):
        scaler = sklearnStandardScaler()
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data
