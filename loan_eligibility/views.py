from django.shortcuts import render
import os
import joblib
import pandas as pd
import xgboost as xgb

# Load your saved Booster model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'loan_model.pkl')
bst = joblib.load(MODEL_PATH)

def home(request):
    return render(request, 'loan_eligibility/home.html')


def predict(request):
    if request.method == 'POST':
        try:
            # Get form inputs
            gender = request.POST.get('gender')
            married = request.POST.get('married')
            dependents = float(request.POST.get('dependents'))
            education = request.POST.get('education')
            self_employed = request.POST.get('self_employed')
            applicant_income = float(request.POST.get('applicant_income'))
            coapplicant_income = float(request.POST.get('coapplicant_income'))
            loan_amount = float(request.POST.get('loan_amount'))
            loan_amount_term = float(request.POST.get('loan_amount_term'))
            credit_history = float(request.POST.get('credit_history'))
            property_area = request.POST.get('property_area')

            # Mapping dropdown/text inputs to numeric values
            gender_map = {'Male': 1, 'Female': 0}
            married_map = {'Yes': 1, 'No': 0}
            education_map = {'Graduate': 1, 'Not Graduate': 0}
            self_employed_map = {'Yes': 1, 'No': 0}
            property_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

            # Create DataFrame with correct column order
            X = pd.DataFrame([[
                gender_map[gender],
                married_map[married],
                dependents,
                education_map[education],
                self_employed_map[self_employed],
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term,
                credit_history,
                property_map[property_area]
            ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

            # Convert to DMatrix (Booster expects this)
            dmatrix = xgb.DMatrix(X)

            # Make prediction
            prediction = bst.predict(dmatrix)[0]
            result = "Eligible " if prediction >= 0.5 else "Not Eligible"

            return render(request, 'loan_eligibility/home.html', {'result': result})

        except Exception as e:
            return render(request, 'loan_eligibility/home.html', {'result': f"Error: {str(e)}"})

    # GET request
    return render(request, 'loan_eligibility/home.html')
