# 🏠 House Price Prediction — Bengaluru

## 📋 Overview
This project predicts house prices in **Bengaluru** using various machine learning algorithms. It explores the dataset, cleans and preprocesses the data, performs exploratory data analysis (EDA), and builds predictive models to estimate property prices based on features such as location, size, number of bedrooms, and square footage.

## 🧩 Project Structure
```
House Price Prediction/
├── House Price Prediction.ipynb   # Main Jupyter Notebook
├── Bengaluru_House_Data.csv       # Dataset used
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies (optional)
```# 🏠 House Price Prediction using Machine Learning

## 📌 Overview
This project is a **Machine Learning–powered web application** that predicts house prices based on various features such as location, square footage, number of bathrooms, balconies, and BHK (number of bedrooms).

The model is trained using **Linear Regression** with preprocessing steps (One-Hot Encoding for categorical data and Standard Scaling for numerical features) wrapped in a **Scikit-Learn Pipeline**.  
A **Flask web app** provides an easy-to-use interface where users can input property details and get instant predictions.

---

## 🚀 Features
- Interactive **Flask web app** with Bootstrap UI  
- **Dynamic dropdown** for selecting location (auto-fetched from the trained pipeline)  
- **Preprocessing handled inside the pipeline** (no manual encoding/scaling)  
- **End-to-end deployment-ready** architecture (pickle + Flask integration)  
- Clean and modern UI using **Bootstrap 5**  

---

## 🧠 Machine Learning Workflow
1. **Data Collection & Cleaning:** Dataset includes features such as `location`, `total_sqft`, `bath`, `balcony`, and `BHK`.  
2. **Feature Engineering:**  
   - `location` → OneHotEncoded  
   - Numerical columns → Scaled using `StandardScaler`  
3. **Model Training:**  
   ```python
   Pipeline([
       ('columntransformer',
           ColumnTransformer(
               transformers=[
                   ('one_hot_encoder',
                       OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False, dtype=int),
                       ['location']
                   )
               ],
               remainder='passthrough'
           )
       ),
       ('standardscaler', StandardScaler()),
       ('linearregression', LinearRegression())
   ])
   ```
4. **Model Saving:** The trained pipeline is serialized as `pipeline.pkl` for deployment.  
5. **Flask Integration:** Loads the trained pipeline, renders input form via `index.html`, and returns predicted price dynamically.

---

## 🧰 Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3 |
| Web Framework | Flask |
| Machine Learning | Scikit-Learn, Pandas, NumPy |
| Frontend | HTML, CSS, Bootstrap 5 |
| Serialization | Pickle |

---

## 🗂️ Project Structure
```
House Price Prediction/
│
├── app.py                 # Flask web app
├── pipeline.pkl           # Trained ML pipeline
├── templates/
│   └── index.html         # Frontend template (Bootstrap)
├── static/                # (optional) CSS, images, etc.
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

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

---

## 🖥️ Usage
1. Select a **location** from the dropdown.  
2. Enter details for **total square feet, bathrooms, balconies, and BHK**.  
3. Click **Predict Price**.  
4. The predicted house price will be displayed in ₹ (Lakh).

---

## 📊 Example Prediction
| Input | Example |
|--------|----------|
| Location | Whitefield |
| Total Sqft | 1200 |
| Bathrooms | 2 |
| Balconies | 1 |
| BHK | 3 |
| **Predicted Price** | ₹ 85.73 Lakh |

---

## 🧩 Future Improvements
- Add more ML models (Random Forest, XGBoost) for better accuracy  
- Deploy on **Render / Vercel / AWS EC2**  
- Add visualizations (price distribution, feature importance)  
- Include an API endpoint for programmatic predictions  

---

## 📚 Dependencies
You can list them in `requirements.txt`:
```
flask
numpy
pandas
scikit-learn
```

---

## 👨‍💻 Author
**Supratim Saha**  
📧 your.email@example.com  
💼 [LinkedIn Profile or GitHub Link]


## ⚙️ Features
- Data cleaning (handling missing values, removing outliers)
- Feature engineering (extracting BHK, total sqft, location encoding)
- Exploratory Data Analysis (EDA) using `matplotlib`, `seaborn`, and `plotly`
- Model training and evaluation
- Model comparison and prediction visualization

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

## 🚀 How to Run
### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook "House Price Prediction.ipynb"
```
## Demo:
<img width="1700" height="860" alt="image" src="https://github.com/user-attachments/assets/6c20ea29-9f7b-412a-91d9-a124007ecbd6" />


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

## 💡 Future Improvements
- Deploy as a web app using **Streamlit** or **Flask**
- Integrate real-time Bengaluru housing data
- Implement cross-validation and hyperparameter tuning
- Build an interactive dashboard for visualization

## 👨‍💻 Author
**Supratim Saha**  
Feel free to connect on [LinkedIn](https://www.linkedin.com/) or [GitHub](https://github.com/).
