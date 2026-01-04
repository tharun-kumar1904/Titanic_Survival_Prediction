# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project builds a machine learning system to predict whether a passenger
survived the RMS Titanic disaster using demographic and travel-related features.
The system covers the complete ML lifecycle including data preprocessing,
feature engineering, exploratory data analysis (EDA), model training, evaluation,
and deployment through an interactive Streamlit web application.

Optional extensions include real-time prediction using Kafka and containerized
deployment using Docker.

---

## 🎯 Objectives
- Predict passenger survival using machine learning models
- Analyze survival trends based on age, gender, class, and fare
- Provide an interactive UI for filtering and sorting predictions
- Demonstrate real-time ML inference using Kafka (optional)
- Ensure reproducibility using Docker (optional)

---

## 📊 Dataset
The dataset is sourced from the **Kaggle Titanic Dataset** and includes the
following features:

- PassengerId  
- Pclass  
- Name  
- Sex  
- Age  
- SibSp  
- Parch  
- Ticket  
- Fare  
- Cabin  
- Embarked  
- Survived (Target Variable)

Raw datasets are stored in:
```
data/raw/
```


Processed features are generated dynamically during preprocessing and training.

---

## 🛠️ Feature Engineering
The following features are engineered to improve model performance:

- **FamilySize** = SibSp + Parch + 1  
- **IsAlone**: Indicates whether the passenger traveled alone  
- **Title Extraction** from passenger names (Mr, Mrs, Miss, Rare)  
- **HasCabin**: Binary indicator for cabin availability  

Categorical variables are encoded using one-hot encoding:
- Sex  
- Embarked  
- Pclass  
- Title  

High-missing or low-utility columns (Ticket, Cabin) are dropped after feature extraction.

---

## 🔍 Exploratory Data Analysis (EDA)
EDA is performed in:
```
notebooks/EDA_Titanic.ipynb
```


### Key Insights
- Female passengers had significantly higher survival rates  
- First-class passengers survived more often  
- Children showed higher survival probability  
- Fare and passenger class strongly influenced survival  
- Strong interaction exists between gender and passenger class  

---

## 🤖 Machine Learning Models
The following models were trained and evaluated:

- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  

---

## 📐 Evaluation Metrics
Models were evaluated using multiple complementary metrics to ensure balanced and reliable performance:

- **Accuracy** – Overall correctness of predictions  
- **Precision** – Reliability of predicted survivors  
- **Recall** – Ability to correctly identify actual survivors  
- **F1-Score** – Balance between precision and recall  
- **Confusion Matrix** – Detailed error analysis  
- **ROC-AUC** – Model discrimination capability  
- **Optimized Probability Threshold** – Tuned to maximize F1-score  

---

## 📈 Model Performance Summary
The final selected model (Logistic Regression with threshold optimization) achieved the following results on the test set:

- **ROC-AUC:** 0.87  
- **Accuracy:** 0.84  
- **Precision:** 0.77  
- **Recall:** 0.83  
- **F1-Score:** 0.80  
- **Optimal Classification Threshold:** 0.52  

The model demonstrates strong generalization and balanced performance across all evaluation metrics.

Saved artifacts:
The final selected model is saved to:
```
models/titanic_model.joblib
models/feature_columns.joblib
models/best_threshold.joblib
```

---
## 🌐 Streamlit Web Application
The Streamlit application enables interactive exploration and prediction.

### Features
- Filter passengers by:
  - Passenger Class  
  - Gender  
  - Age Range  
- View predicted survival probabilities  
- Sort passengers based on risk levels  
- Explore survival patterns through interactive visualizations  

### ▶️ Run Streamlit App

```bash
https://titanic-ml-survival-predictor.streamlit.app/
```

---
## 🔄 Kafka – Real-Time Prediction (Optional)
Kafka is used to simulate real-time passenger data streaming.

### Components:
- **Producer:** Streams passenger data row-by-row
- **Consumer:** Applies preprocessing and predicts survival probability

```bash
python src/kafka/consumer.py
python src/kafka/producer.py
```

Kafka topic used:
```
titanic_passengers
```

---

## 🐳 Docker Deployment (Optional)

Build Docker image:
```bash
docker build -t titanic-app -f docker/Dockerfile .
```

Run container:
```bash
docker run -p 8501:8501 titanic-app
```

---

## 📁 Project Structure
```
Titanic_Survival_Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── EDA_Titanic.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── kafka/
│       ├── producer.py
│       └── consumer.py
├── models/
│   ├── titanic_model.joblib
│   ├── feature_columns.joblib
│   └── best_threshold.joblib
├── streamlit_app/
│   └── app.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🎥 Video Demonstration
A 5–10 minute video demonstration includes:
- Project overview
- EDA walkthrough
- Model training & evaluation
- Streamlit UI demo
- Kafka real-time prediction (optional)

(Video link to be added)

---

## 🧰 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Kafka
- Docker

---

Youtube Link : https://youtu.be/v39DD87l7Q0

## ✅ Final Notes
This project follows clean coding practices, modular design, and reproducible
machine learning workflows aligned with real-world ML deployment standards.
Emphasis is placed on ethical evaluation, interpretability, and usability.

---

## ▶️ How to Run the Project

### Step 1:
```bash
pip install -r requirements.txt
```

### Step 2:
```bash
cd src
python train_model.py
```

### Step 3:
```bash
cd ../streamlit_app
streamlit run app.py
```
