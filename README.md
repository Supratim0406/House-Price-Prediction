# 🏠 House Price Prediction — Bengaluru

## 📋 Overview
A machine learning–powered House Price Prediction web app built using Python, Flask, Pandas, Scikit-Learn, and Bootstrap.
Users can enter property details (location, sqft, BHK, bath, balcony), and the app will predict the estimated house price using a trained ML model.

This project predicts house prices in **Bengaluru** using various machine learning algorithms. It explores the dataset, cleans and preprocesses the data, performs exploratory data analysis (EDA), and builds predictive models to estimate property prices based on features such as location, size, number of bedrooms, and square footage.

## 📌 Features

* 🧠 ML Model (Pipeline) trained using scikit-learn
* 🌍 Location dropdown auto-loaded from cleaned dataset
* 🌐 Interactive Flask Web Interface
* 🎨 Modern UI (Bootstrap)
* 📝 Cleaned dataset used for dynamic dropdown
* 📊 Model saved as model.pkl for inference

## 🧠 Machine Learning Models
The notebook typically includes or can include:
- **Linear Regression**
- **Lasso / Ridge Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **XGBoost / Gradient Boosting**
- Performance evaluation using **R² Score**, **MAE**, or **RMSE**

## 📊 Exploratory Data Analysis (EDA)
Key insights explored:
- Distribution of house prices across locations  
- Relation between size (sqft) and price  
- Price per square foot by location  
- Correlation between features  

## 🧹 Data Preprocessing
Steps performed:
1. Handling missing values  
2. Removing duplicate and irrelevant columns  
3. Converting text-based features (like “2 BHK”) to numerical form  
4. Encoding categorical variables (e.g., one-hot encoding for location)  
5. Feature scaling (if required)  


## 🧩 Project Structure
```
House Price Prediction/
│
├── app.py                 # Flask web application
├── model.pkl              # Trained ML pipeline
├── Cleaned_data.csv       # Pre-processed dataset
├── templates/
│   └── index.html         # Front-end UI
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

```

## 🚀 How It Works

User selects location from dropdown
User enters
Total Sqft
Number of Bathrooms
Balcony
BHK
Flask sends the input to the ML model
Model returns the predicted price
Result is displayed on the webpage


## 🧰 Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3 |
| Web Framework | Flask |
| Machine Learning | Scikit-Learn, Pandas, NumPy |
| Frontend | HTML, CSS, Bootstrap 5 |
| Serialization | Pickle |


## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask App
```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

## 🖥️ Usage
1. Select a **location** from the dropdown.  
2. Enter details for **total square feet, bathrooms, balconies, and BHK**.  
3. Click **Predict Price**.  
4. The predicted house price will be displayed in ₹ (Lakh).


## 📊 Example Prediction
| Input | Example |
|--------|----------|
| Location | Whitefield |
| Total Sqft | 1200 |
| Bathrooms | 2 |
| Balconies | 1 |
| BHK | 3 |
| **Predicted Price** | ₹ 85.73 Lakh |



## 🧩 Future Improvements
- Add more ML models (Random Forest, XGBoost) for better accuracy  
- Deploy on **Render / Vercel / AWS EC2**  
- Add visualizations (price distribution, feature importance)  
- Include an API endpoint for programmatic predictions  

## 📚 Dependencies
You can list them in `requirements.txt`:
```
flask
numpy
pandas
scikit-learn
```

## Demo:
<img width="1644" height="913" alt="image" src="https://github.com/user-attachments/assets/9aa0735d-065c-4491-a90b-b0cbd413c3fb" />

## 🧾 Requirements
Example `requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
xgboost
```

## 📈 Results
The model outputs predicted house prices for given input features and provides a visual comparison between actual and predicted values.  
You can tweak hyperparameters or try other models to improve prediction accuracy.


## 👨‍💻 Author
**Supratim Saha**  
Feel free to connect on [LinkedIn](https://www.supratimsmail.com/) or [GitHub](https://github.com/Supratim0406).
