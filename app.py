from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load pre-trained random forest model and preprocessors (train and store as shown before)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        iq = float(request.form['IQ'])
        prev_sem = float(request.form['PrevSemResult'])
        cgpa = float(request.form['CGPA'])
        academic = float(request.form['AcademicPerformance'])
        internship = request.form['InternshipExperience']
        extra_curricular = float(request.form['ExtraCurricularScore'])
        communication = float(request.form['CommunicationSkills'])
        projects = float(request.form['ProjectsCompleted'])

        internship_encoded = label_encoder.transform([internship])[0]
        features = np.array([[iq, prev_sem, cgpa, academic, internship_encoded,
                              extra_curricular, communication, projects]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prediction = 'Eligible for Placement' if pred == 1 else 'Not Eligible for Placement'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
