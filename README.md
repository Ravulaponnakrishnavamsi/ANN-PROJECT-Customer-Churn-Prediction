# COUSTMER CHURN PREDICTION 
### This is an interactive Streamlit web application that predicts whether a bank customer is likely to churn (leave the bank) based on their personal and financial details.
### The app uses a trained TensorFlow deep learning model, along with saved preprocessing tools (LabelEncoder, OneHotEncoder, and StandardScaler) to transform user inputs into model-ready features.'

 ### FOLDER STRUCTURE
 ```
├── app.py                     # Streamlit application code
├── model.h5                   # Trained Keras/TensorFlow model
├── label_encoder.pkl          # Label encoder for Gender
├── onehot_encoder_geo.pkl     # One-hot encoder for Geography
├── scaler.pkl                 # StandardScaler for numeric features
├── README.md                  # Documentation
└── requirements.txt           # Python dependencies
```
Installation & Setup

Clone the repository
```
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

Install dependencies
```
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:
```
bash
``` Copy code
pip install streamlit tensorflow scikit-learn pandas numpy
Run the app
```
bash
```
streamlit run app.py
```
### The app will open in your browser at:

👉 [http://localhost:8501/]

# Example Prediction
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
```
Churn Probability: 0.18
The customer is not likely to churn.
```
