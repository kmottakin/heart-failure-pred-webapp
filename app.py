from flask import Flask, render_template, request
import pickle
import numpy as np

with open(f'model/heart_failure_model_RF.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['anaemia']
    data3 = request.form['creatinine_phosphokinase']
    data4 = request.form['diabetes']
    data5 = request.form['ejection_fraction']
    data6 = request.form['high_blood_pressure']
    data7 = request.form['platelets']
    data8 = request.form['serum_creatinine']
    data9 = request.form['serum_sodium']
    data10 = request.form['sex']
    data11 = request.form['smoking']
    data12 = request.form['time']

    anaemia_list = {'Yes': 1,'No': 0}
    diabetes_list = {'Yes': 1,'No': 0}
    high_blood_pressure_list = {'Yes': 1,'No': 0}
    sex_list = {'Male': 1,'Female': 0}
    smoking_list = {'Yes': 1,'No': 0}
    arr = np.array([[data1, anaemia_list[data2], data3, diabetes_list[data4], data5, high_blood_pressure_list[data6], data7, data8, data9, sex_list[data10], smoking_list[data11], data12]])
    pred = model.predict_proba(arr)
    #print(pred)
    pred = pred[:, 1]
    if(pred >= 0.35):
        final_pred = 1
    else:
        final_pred = 0
    return render_template('after.html', data=final_pred)


if __name__ == "__main__":
    app.run(debug=True)