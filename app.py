from flask import Flask, render_template, request
import pandas as pd
import pickle


# -------------------------------
# Load model and cleaned dataset
# -------------------------------

model_pipeline = pickle.load(open('model.pkl', 'rb'))

clean_df = pd.read_csv("Cleaned_data.csv")
locations = sorted(clean_df['location'].unique())   # unique list for dropdown

app = Flask(__name__)

# -------------------------------
# Prediction function
# -------------------------------

def predict_house_price(model_pipeline, location, total_sqft, bath, balcony, BHK):
    input_data = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony],
        'BHK': [BHK]
    })

    prediction = model_pipeline.predict(input_data)
    return prediction[0]


# -------------------------------
# Flask routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None

    if request.method == 'POST':
        location = request.form['location']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        BHK = int(request.form['BHK'])

        predicted_price = predict_house_price(
            model_pipeline,
            location,
            total_sqft,
            bath,
            balcony,
            BHK
        )

    return render_template(
        'index.html',
        predicted_price=predicted_price,
        locations=locations
    )


# -------------------------------
# Run the app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)