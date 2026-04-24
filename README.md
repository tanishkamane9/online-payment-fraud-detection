# 💳 Online Payment Fraud Detection System

A Machine Learning-based web application that detects fraudulent online transactions in real-time using **XGBoost** and **Flask**.

---

## 🚀 Project Overview

This project focuses on identifying fraudulent financial transactions using advanced machine learning techniques.
Due to the highly imbalanced nature of fraud datasets, special attention is given to **recall optimization** to ensure maximum fraud detection.

---

## 🧠 Key Features

* 🔍 Detects fraudulent vs non-fraudulent transactions
* ⚡ Real-time prediction using Flask web app
* 📊 Handles imbalanced dataset effectively
* 🔧 Feature engineering and preprocessing applied
* 🤖 Model optimized using XGBoost

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Flask
* HTML/CSS

---

## 📂 Project Structure

```
.
├── app.py                         # Flask application
├── fraud_detection_model.pkl      # Trained XGBoost model
├── templates/                     # HTML templates
│   ├── home.html
│   ├── predict.html
│   └── submit.html
├── static/                        # Images and assets
├── online_payment_fraud_detection.ipynb   # Jupyter Notebook (EDA + Training)
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. User enters transaction details in the web interface
2. Data is preprocessed (log transformation, encoding, etc.)
3. Trained XGBoost model predicts fraud probability
4. Result is displayed as:

```
Fraud 🚨  OR  Not Fraud ✅
```

---

## 📊 Model Performance

* ✅ Accuracy: ~99%
* 🎯 Recall (Fraud Detection): ~77%
* 🔥 Precision: ~99%

> Note: Recall is prioritized to reduce missed fraud cases.

---

## 🧪 Preprocessing Steps

* Log transformation applied to skewed features (`amount`)
* Label encoding for categorical feature (`type`)
* Feature selection and cleaning
* Handling of imbalanced dataset

---

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/online-payment-fraud-detection.git
cd online-payment-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask app

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 💡 Future Improvements

* Improve recall beyond 80% using advanced tuning
* Add probability/confidence score in UI
* Deploy on cloud (Render / AWS)
* Add user authentication system

---

## 🧾 Conclusion

XGBoost provided the best balance between precision and recall among all tested models, making it highly effective for fraud detection in imbalanced datasets.

---

## 👩‍💻 Author

**Tanu Mane**
Computer Science Engineer | Aspiring AI Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
