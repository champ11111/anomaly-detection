import json
from datetime import datetime

from airflow import DAG
from airflow.hooks.mysql_hook import MySqlHook
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import requests

import os
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

def get_traffy_data():
    #baseurl = localhost port 9000
    base_url = 'http://host.docker.internal:9000/traffy_data'

    response = requests.get(base_url)

    input_json = response.json()

    # #Write to file named "traffy_data_{start_date}_{end_date}.txt in folder data"
    # with open(f'traffy_data_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.txt', 'w') as f:
    #     f.write("%s\n" % input_json)
    
    return input_json

def predict_anomaly(**context):
    # Access X from the previous task using XCom
    input_json = context['task_instance'].xcom_pull(task_ids='get_traffy_data')

    json_object = json.dumps(input_json, indent = 4) 
    # Specify the URL you want to send the POST request to
    url = "http://host.docker.internal:8081/invocations"

    # Set the content type header to indicate JSON data
    headers = {"Content-Type": "application/json"}

    # Send the POST request to the invocations endpoint
    response = requests.post(url, data=json_object, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # Request was successful
        response_data = response.json()  # Assuming the response is in JSON format
        print("Response data:", response_data)
    else:
        # Request was not successful
        print("Request failed with status code:", response.status_code)

    base_url = 'http://host.docker.internal:9000/anomaly'
    data = {"district_list": district_list, "anomaly_list": response_data['predictions']}
    json_data = json.dumps(data)
    response = requests.post(base_url,data=json_data,headers=headers)

    return response.status_code

default_args = {
    'owner': 'dataength',
    'start_date': datetime(2020, 7, 1),
}
with DAG('Anomaly_Detection_Pipeline',
         schedule_interval='@monthly',
         default_args=default_args,
         description='A anomaly detection pipeline',
         catchup=False) as dag:

    get_data_task = PythonOperator(
        task_id='get_traffy_data',
        python_callable=get_traffy_data,
        dag=dag
    )

    predict_task = PythonOperator(
        task_id='predict_anomaly',
        python_callable=predict_anomaly,
        provide_context=True,  # To pass the context to the function
        dag=dag
    )   

    # Define task dependencies
    get_data_task >> predict_task
        