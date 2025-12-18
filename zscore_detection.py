import pandas as pd
import numpy as np

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

data = pd.read_csv("KDDTrain+.txt", names=columns)

# Drop useless column
data.drop("difficulty", axis=1, inplace=True)

# Convert labels
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)

# Select numeric columns only
numeric_data = data.select_dtypes(include=[np.number])

# Z-score calculation
z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())

# Mark anomalies
threshold = 3
anomalies = (z_scores > threshold).any(axis=1)

data["zscore_anomaly"] = anomalies.astype(int)

print(data["zscore_anomaly"].value_counts())
from sklearn.metrics import classification_report

print(
    classification_report(
        data["label"],
        data["zscore_anomaly"]
    )
)