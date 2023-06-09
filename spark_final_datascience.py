# -*- coding: utf-8 -*-
"""spark Final_DataScience.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12kUtTh_G5dvPCI8NLxpSaWxJIy5Te4fb
"""

#Import the libraries
import json
import mlflow
import sys
import mlflow.sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from mlflow.models.signature import infer_signature
import os
base_url = 'https://publicapi.traffy.in.th/share/teamchadchart/search'
state_type = 'finish'
dates = pd.date_range(start='2023-04-18', end='2023-05-18')
# request data from API using query params
def get_data(state_type, start_date, end_date):
    params = {
        'state_type': state_type,
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d')
    }
    response = requests.get(base_url, params=params)
    return response.json()
data = []
for i in tqdm(range(len(dates)-1)):
    data += get_data(state_type, dates[i], dates[i+1])["results"]

df = pd.DataFrame(data)


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, udf, count, unix_timestamp
from pyspark.sql.types import StringType
from pyspark.ml.feature import StandardScaler, VectorAssembler

import matplotlib.pyplot as plt

os.environ["SPARK_HOME"] = "/opt/homebrew/Cellar/apache-spark/3.4.0/libexec"
import findspark
findspark.init()

spark_url = 'local'

spark = SparkSession.builder\
        .master(spark_url)\
        .appName('Spark SQL')\
        .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Convert pandas DataFrame to PySpark DataFrame
spark_df = spark.createDataFrame(df)

# Define UDF to check existence of district
def check_existence(address):
    district_list = ['คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
       'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
    'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ',
       'บางขุนเทียน', 'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด',
       'บางรัก', 'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ',
       'ป้อมปราบศัตรูพ่าย', 'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ',
       'มีนบุรี', 'ยานนาวา', 'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว',
       'วังทองหลาง', 'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร',
       'สายไหม', 'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง']
    for word in district_list:
        if word in address:
            return word
    return None

check_existence_udf = udf(check_existence, StringType())

# Apply UDF to create district column
spark_df = spark_df.withColumn("district", check_existence_udf(col("address")))

# Drop rows with missing district values
spark_df = spark_df.dropna(subset=['district'])
spark_df.dropDuplicates(['description'])

spark_df = spark_df.withColumn("time_spend_sec", (unix_timestamp(col("last_activity")) - unix_timestamp(col("timestamp"))).cast("double"))

spark_df.show()
# Group by district and count the occurrences
grouped_df = spark_df.groupBy('district').agg(count('*').alias('count'),avg('time_spend_sec').alias('average_time_spend'))
grouped_df.show()

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
X2 = scaledData.select(["district","time_spend_scaled","average_time_spend",'count']).collect()
# print(X2)
output_df = pd.DataFrame(X2,
                   columns=["district","time_spend_scaled","average_time_spend",'count'])
output_df.to_csv('output.csv', index=False)
epsilon = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9 # Define the maximum distance between samples for clustering
min_samples = float(sys.argv[2]) if len(sys.argv) > 2 else 2 # Minimum number of samples required to form a dense region

class MyDBSCAN(DBSCAN):
    def predict(self, X):
        self.fit(X)
        return self.labels_


dbscan = MyDBSCAN(eps=epsilon, min_samples=min_samples)



labels = dbscan.predict(X)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

import matplotlib.pyplot as plt

# Visualize the DBSCAN clusters
plt.scatter(X,[0] * len(X), c=labels)
plt.xlabel('Time Spend (Scaled)')
plt.ylabel('Count (Scaled)')
plt.title('DBSCAN Clustering Results')
plt.show()



import requests

# Specify the URL you want to send the POST request to
url = "http://0.0.0.0:8080/invocations"




# Convert input data to a JSON-serializable format
input_json = json.dumps({"inputs": [list(e) for e in X]})

# Set the content type header to indicate JSON data
headers = {"Content-Type": "application/json"}

# Send the POST request to the invocations endpoint
response = requests.post(url, data=input_json, headers=headers)

# Check the response status code
if response.status_code == 200:
    # Request was successfulß
    response_data = response.json()  # Assuming the response is in JSON format
    print("Response data:", response_data)

else:
    # Request was not successful
    print("Request failed with status code:", response.status_code)

