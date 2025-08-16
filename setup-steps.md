# Create virtual environment
python3 -m venv ml_api_env

# Activate the virtual environment
source ml_api_env/bin/activate

# Install basic packages
pip install fastapi uvicorn scikit-learn pandas pydantic joblib pytest ruff black kaggle

# Or install from requirements.txt (if you have one)
pip install -r requirements.txt

# List installed packages
pip list

# Save current packages to requirements.txt
pip freeze > requirements.txt

# Data Generators

# 1) Regression (good for Linear/Ridge/RandomForestRegressor)
python synthetic_data_generator.py --algo linear_regression --n-samples 1000 --n-features 12 --noise 8 --out data/reg.csv

# 2) Binary classification (Logistic/SVM/RF)
python synthetic_data_generator.py --algo logistic_regression --n-samples 2000 --n-features 10 --class-sep 1.5 --weights 0.7,0.3 --out data/cls.csv

# 3) Multi-class classification
python synthetic_data_generator.py --algo svm_classifier --n-samples 1500 --n-features 6 --classes 4 --out data/cls_multi.csv

# 4) Clustering (KMeans/DBSCAN/GMM)
python synthetic_data_generator.py --algo kmeans --n-samples 1200 --n-features 3 --clusters 5 --cluster-std 0.9 --out data/kmeans.csv

# 5) Time-series (sales forecasting)
python synthetic_data_generator.py --algo time_series --days 730 --stores 20 --items 5 --promo-prob 0.25 --out data/ts_sales.csv
