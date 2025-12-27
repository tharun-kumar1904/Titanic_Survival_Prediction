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
- **Title Extraction** from names (Mr, Mrs, Miss, Rare)  
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

### Key Insights:
- Females had significantly higher survival rates
- First-class passengers survived more often
- Children had higher survival probability
- Fare and passenger class strongly influenced survival
- Strong interaction exists between gender and passenger class

---

## 🤖 Machine Learning Models
The following models are trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Threshold tuning for F1 optimization

The final selected model is saved to:
```
models/titanic_model.joblib
```

---

## 📈 Model Performance Summary
The best-performing model (Logistic Regression) achieved:

- **ROC-AUC:** ~0.87  
- **Accuracy:** ~0.84  
- **F1-Score:** ~0.80  

### 🔎 Why performance plateaus below 90%
The Titanic dataset is small and noisy, with many unobserved factors influencing survival.
Without introducing data leakage or rule-based heuristics, performance naturally plateaus
around this range. The project prioritizes **ethical evaluation, generalization, and
real-world reliability** over artificially inflated metrics.

---

## 🌐 Streamlit Web Application
The Streamlit app enables users to:

- Filter passengers by:
  - Passenger Class
  - Gender
  - Age Range
- View predicted survival probabilities
- Sort passengers based on risk levels
- Explore survival patterns interactively

### ▶️ Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```

Access the app at:
```
http://localhost:8501
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

Access:
```
http://localhost:8501
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
