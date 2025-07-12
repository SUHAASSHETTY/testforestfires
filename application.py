import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scalar pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler=pickle.load(open('models/scaler.pkl', 'rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == "POST":
        # Handle the POST request (form submission)
        # Extract input data and make predictions
        temperature = request.form['Temperature']
        rh = request.form['RH']
        ws = request.form['Ws']
        rain = request.form['Rain']
        ffmc = request.form['FFMC']
        dmc = request.form['DMC']
        isi = request.form['ISI']
        classes = request.form['Classes']
        region = request.form['Region']
        
        # Here, you'll have your model prediction logic (example):
        # You would need to apply any necessary preprocessing using the standard_scaler, 
        # then use the ridge_model to predict the result.

        data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        scaled_data = standard_scaler.transform(data)  # This assumes the scaler was trained with this kind of data
        prediction = ridge_model.predict(scaled_data)

        result = prediction[0]  # Just for example, depending on the model output

        return render_template('home.html', result=result)

    else:
        return render_template('home.html')




if __name__=="__main__":
    app.run(host="0.0.0.0")