from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import requests

import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, count, unix_timestamp
from pyspark.sql.types import StringType
from pyspark.ml.feature import StandardScaler, VectorAssembler

import findspark

district_list = ['คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
        'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
        'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ',
        'บางขุนเทียน', 'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด',
        'บางรัก', 'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ',
        'ป้อมปราบศัตรูพ่าย', 'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ',
        'มีนบุรี', 'ยานนาวา', 'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว',
        'วังทองหลาง', 'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร',
        'สายไหม', 'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง']


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/traffy_data")
def get_traffy_data():
    base_url = 'https://publicapi.traffy.in.th/share/teamchadchart/search'
    state_type = 'finish'

    def get_data(state_type, start_date, end_date):
        params = {
            'state_type': state_type,
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
        response = requests.get(base_url, params=params)
        return response.json()

    # Get today's date
    end_date = datetime.today()

    # Calculate the start date as one month before today
    start_date = end_date - timedelta(days=30)

    # Create the date range
    dates = pd.date_range(start=start_date, end=end_date)

    data = []
    for i in tqdm(range(len(dates)-1)):
        data += get_data(state_type, dates[i], dates[i+1])["results"]

    df = pd.DataFrame(data)

    findspark.init()

    spark_url = 'local'

    spark = SparkSession.builder\
            .master(spark_url)\
            .appName('Spark SQL')\
            .getOrCreate()

    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # Convert pandas DataFrame to PySpark DataFrame
    spark_df = spark.createDataFrame(df)

    def check_existence(address):
        for word in district_list:
            if word in address:
                return word
        return None

    # Define UDF to check existence of district
    check_existence_udf = udf(check_existence, StringType())

    # Apply UDF to create district column
    spark_df = spark_df.withColumn("district", check_existence_udf(col("address")))

    # Drop rows with missing district values
    spark_df = spark_df.dropna(subset=['district'])
    spark_df = spark_df.withColumn("time_spend_sec", (unix_timestamp(col("last_activity")) - unix_timestamp(col("timestamp"))).cast("double"))

    # Group by district and count the occurrences
    grouped_df = spark_df.groupBy('district').agg(count('*').alias('count'),avg('time_spend_sec').alias('average_time_spend'))

    assembler = VectorAssembler(inputCols=["average_time_spend"], outputCol="average_time_spend_vec")
    grouped_df = assembler.transform(grouped_df)
    # Define the scaler
    scaler = StandardScaler(inputCol="average_time_spend_vec", outputCol="time_spend_scaled", withStd=True, withMean=True)

    # Fit the scaler to the data
    scalerModel = scaler.fit(grouped_df)

    # Apply the scaler to transform the data
    scaledData = scalerModel.transform(grouped_df)

    # Show the transformed data
    scaledData.show()

    X = scaledData.select("time_spend_scaled").rdd.flatMap(lambda x: x).collect()
    
    output_json = {"inputs": [list(e) for e in X]}

    
    district_list_spark = scaledData.select("district").rdd.flatMap(lambda x: x).collect()
    print(district_list)

    district_list = [e for e in district_list_spark]

    print(district_list)
    return output_json

class Item(BaseModel):
    district_list: list = None
    anomaly_list: list = None

@app.post("/anomaly")
async def create_item(item: Item):
    # Process the item and perform necessary operations
    df = pd.DataFrame({'district': item.district_list, 'anomaly': item.anomaly_list})
    df.to_csv(f'anomaly_data/anomaly_data_{datetime.today().strftime("%Y-%m-%d")}.csv', index=False)
    # Return a response
    return {"message": "File created successfully"}