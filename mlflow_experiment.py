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
import pickle

class MyDBSCAN(DBSCAN):
    def predict(self, X):
        self.fit(X)
        return self.labels_



base_url = 'https://publicapi.traffy.in.th/share/teamchadchart/search'
state_type = 'finish'
dates = pd.date_range(start='2022-05-19', end='2023-05-19')
# request data from API using query params
mlflow.set_tracking_uri("http://127.0.0.1:5000")
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



# file = open('traffy', 'wb')

# # dump information to that file
# pickle.dump(data, file)

# # close the file
# file.close()

# file = open('traffy', 'rb')

# # dump information to that file
# data = pickle.load(file)

# # close the file
# file.close()
df = pd.DataFrame(data)

df.drop_duplicates(subset=['description'], keep='first', inplace=True)
drop_list = ["ticket_id","type","org","coords","photo_url","after_photo","state","star","count_reopen","description","problem_type_abdul","note"]
df.drop(drop_list, axis=1, inplace=True)

df['last_activity'] = pd.to_datetime(df['last_activity'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time_spend'] = df['last_activity'] - df['timestamp']

district_list = ['คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
       'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
       'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ',
       'บางขุนเทียน', 'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด',
       'บางรัก', 'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ',
       'ป้อมปราบศัตรูพ่าย', 'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ',
       'มีนบุรี', 'ยานนาวา', 'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว',
       'วังทองหลาง', 'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร',
       'สายไหม', 'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง']

def check_existence(address, district_list):
    try:
      for word in district_list:
        if word in address:
          return word
      return None
    except TypeError:
        return 0

df['district'] = df.apply(lambda x: check_existence(x.address,district_list), axis=1)



df.dropna(subset=['district'],inplace=True)

# Group by district and count the occurrences
df['time_spend_sec'] = df['time_spend'].dt.total_seconds()


# Calculate the average time spend
grouped_df = df.groupby('district').size().to_frame("count")
grouped_df['average_time_spend'] = df.groupby('district')['time_spend_sec'].mean()


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

scaler = StandardScaler()
grouped_df['time_spend_scaled'] = scaler.fit_transform(grouped_df['average_time_spend'].values.reshape(-1, 1))

# Start MLflow run
min_samples = 2 # Minimum number of samples required to form a dense region

start = 0.05
stop = 0.9
step = 0.05

EXPERIMENT_NAME = "dbscan-experiment4"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

for idx, eps in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1, 1.2, 1.3,1.4,1.5]):
    dbscan = MyDBSCAN(eps=eps, min_samples=min_samples)
    X = grouped_df[['time_spend_scaled']].values
    labels = dbscan.predict(X)
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        if n_clusters_ == 1 and n_noise_ == 0:
            print(f"Silhouette Coefficient: 0")
        else:
            print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
        # Log parameters
        mlflow.log_param("epsilon", eps)
        # Log metrics
        mlflow.log_metric("Estimated number of clusters", n_clusters_)
        mlflow.log_metric("Estimated number of noise points", n_noise_)
        
        if n_clusters_ == 1 and n_noise_ == 0:
           mlflow.log_metric("Silhouette Coefficient", 0)
        else:
            mlflow.log_metric("Silhouette Coefficient", metrics.silhouette_score(X, labels))
        # Log the sklearn model 
        signature = infer_signature(X, labels)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                dbscan, "DBSCAN_Model", registered_model_name="DBSCAN_Model", signature=signature
            )
        else:
            mlflow.sklearn.log_model(dbscan, "DBSCAN_Model", signature=signature)
        