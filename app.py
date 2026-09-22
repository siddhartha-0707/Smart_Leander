from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        gender = int(request.form['gender'])
        own_car = int(request.form['own_car'])
        own_realty = int(request.form['own_realty'])
        income_type = int(request.form['income_type'])
        education = int(request.form['education'])
        family_status = int(request.form['family_status'])
        housing_type = int(request.form['housing_type'])
        occupation = int(request.form['occupation'])
        annual_income = float(request.form['annual_income'])
        age = int(request.form['age'])
        employment_years = float(request.form['employment_days'])
        family_members = int(request.form['family_members'])

        
        days_birth = int(-age * 365.25)
        
        
        if employment_years == 0:
            days_employed = 365243
        else:
            days_employed = int(-employment_years * 365.25)

        
        features = np.array([[
            gender, own_car, own_realty, income_type,
            education, family_status, housing_type,
            occupation, annual_income, days_birth,
            days_employed, family_members
        ]])

        
        prediction = model.predict(features)[0]
        result = 'APPROVED' if prediction == 0 else 'REJECTED'
        color = 'green' if prediction == 0 else 'red'
        
        
        details = {
            'annual_income': annual_income,
            'age': age,
            'employment': employment_years,
            'family_members': family_members,
            'gender': gender,
            'own_car': own_car,
            'own_realty': own_realty,
            'income_type': income_type
        }

        return render_template('result.html', result=result, color=color, details=details)
        
    except Exception as e:
        return f'<div style="color:red; font-family:sans-serif; padding:2rem;"><h3>Error Processing Request</h3><p>{str(e)}</p></div>'

if __name__ == '__main__':
    app.run(debug=True)
