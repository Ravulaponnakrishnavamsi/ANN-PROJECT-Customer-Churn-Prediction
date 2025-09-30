Customer Churn Prediction App

This is an interactive Streamlit web application that predicts whether a bank customer is likely to churn (leave the bank) based on their personal and financial details.

The app uses a trained TensorFlow deep learning model, along with saved preprocessing tools (LabelEncoder, OneHotEncoder, and StandardScaler) to transform user inputs into model-ready features.

✨ Features

User-friendly interface built with Streamlit.

Collects customer details:

Geography

Gender

Age

Balance

Credit Score

Estimated Salary

Tenure

Number of Products

Credit Card ownership

Active Member status

Preprocessing pipeline:

Label encoding for categorical gender.

One-hot encoding for geography.

Standard scaling for numerical features.

Prediction output:

Churn probability (0–1).

Clear interpretation: Likely to churn / Not likely to churn.

🛠️ Tech Stack

Frontend: Streamlit

Backend / Model: TensorFlow (Keras Sequential model)

Preprocessing: scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler)

Data Handling: Pandas, NumPy

Model Persistence: Pickle

📂 Project Structure
├── app.py                     # Streamlit application code
├── model.h5                   # Trained Keras/TensorFlow model
├── label_encoder.pkl          # Label encoder for Gender
├── onehot_encoder_geo.pkl     # One-hot encoder for Geography
├── scaler.pkl                 # StandardScaler for numeric features
├── README.md                  # Documentation
└── requirements.txt           # Python dependencies

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


Install dependencies

pip install -r requirements.txt


If you don’t have a requirements.txt, install manually:

pip install streamlit tensorflow scikit-learn pandas numpy


Run the app

streamlit run app.py


The app will open in your browser at:
👉 http://localhost:8501/

🧠 Model Training (Optional)

If you want to retrain the churn model and regenerate the preprocessing objects:

Dataset
Use the Churn Modelling dataset
 (or a similar one).

Example columns:

RowNumber, CustomerId, Surname

Geography, Gender, Age, Balance, CreditScore, EstimatedSalary

Tenure, NumOfProducts, HasCrCard, IsActiveMember, Exited

Preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Encode Gender
label_encoder_gender = LabelEncoder()
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])
pickle.dump(label_encoder_gender, open('label_encoder.pkl', 'wb'))

# One-hot encode Geography
onehot_encoder_geo = OneHotEncoder(drop='first')
geo_encoded = onehot_encoder_geo.fit_transform(df[['Geography']]).toarray()
pickle.dump(onehot_encoder_geo, open('onehot_encoder_geo.pkl', 'wb'))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open('scaler.pkl', 'wb'))


Model Training

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2)

model.save('model.h5')

 Example Prediction

Input:

Geography: France

Gender: Male

Age: 42

Credit Score: 650

Balance: 50,000

Estimated Salary: 60,000

Tenure: 5 years

Products: 2

Credit Card: Yes

Active Member: Yes

Output:

Churn Probability: 0.18
The customer is not likely to churn.

✅ Next Steps / Improvements

Add SHAP/LIME interpretability for feature impact.

Save predictions to a database.

Deploy app on Streamlit Cloud / Heroku / AWS.

Enhance model with ensemble methods.

👨‍💻 Author

Developed by Ravula Ponna krishnavamsi.
Feel free to contribute or raise issues.
