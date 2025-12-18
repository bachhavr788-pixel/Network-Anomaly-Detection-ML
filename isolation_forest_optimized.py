import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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

# Load data
data = pd.read_csv("KDDTrain+.txt", names=columns)
data.drop("difficulty", axis=1, inplace=True)
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)

# Numeric features
X = data.select_dtypes(include=[np.number]).drop("label", axis=1)

# SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CONTAMINATION ≈ real anomaly ratio
contamination_rate = data["label"].mean()

model = IsolationForest(
    n_estimators=200,
    contamination=contamination_rate,
    random_state=42
)

model.fit(X_scaled)

pred = model.predict(X_scaled)
pred = np.where(pred == -1, 1, 0)

print(classification_report(data["label"], pred))