# Network Anomaly Detection using Machine Learning

This project implements an unsupervised network anomaly detection system using the NSL-KDD dataset.

## Methods Used
- Z-score based statistical anomaly detection
- Isolation Forest for machine learning-based anomaly detection
- Optimized Isolation Forest using feature scaling and data-driven contamination

## Results
- Z-score: Precision 0.55, Recall 0.21
- Isolation Forest (raw): Precision 0.67, Recall 0.29
- Isolation Forest (optimized): Precision 0.62, Recall 0.62

## Dataset
NSL-KDD dataset  
Source: University of New Brunswick

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Outcome
The optimized Isolation Forest model demonstrated balanced anomaly detection performance, making it suitable for real-world network intrusion detection scenarios.
