import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Multiple Disease Prediction", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
.title-text { 
    font-size: 3rem; font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding: 20px 0;
}
.subtitle { text-align: center; color: #888; font-size: 1.2rem; margin-bottom: 30px; }
.result-box-danger {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    padding: 20px; border-radius: 15px; text-align: center;
    font-size: 1.5rem; font-weight: bold; color: white; margin-top: 20px;
}
.result-box-success {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 20px; border-radius: 15px; text-align: center;
    font-size: 1.5rem; font-weight: bold; color: white; margin-top: 20px;
}
.tip-box {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #00d2ff;
    padding: 15px; border-radius: 10px; margin-top: 15px; color: #ccc;
}
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px; border-radius: 10px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load models
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
heart_scaler = pickle.load(open('heart_scaler.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb'))
parkinsons_scaler = pickle.load(open('parkinsons_scaler.pkl', 'rb'))

# Title
st.markdown('<p class="title-text">🏥 Multiple Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Powered Early Disease Detection Using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
st.sidebar.title("Navigation")
disease = st.sidebar.selectbox("🔍 Select Disease", ["Diabetes", "Heart Disease", "Parkinsons"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.success("🩸 Diabetes: 72.08%")
st.sidebar.warning("❤️ Heart Disease: 98.54%")
st.sidebar.info("🧠 Parkinsons: 89.74%")

# =================== DIABETES ===================
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    st.markdown("Enter patient details below to predict diabetes risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose Level", 0, 300)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", 0, 150)
        SkinThickness = st.number_input("Skin Thickness (mm)", 0, 100)
    with col2:
        Insulin = st.number_input("Insulin Level", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        Age = st.number_input("Age", 1, 120)

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)
        probability = diabetes_model.predict_proba(input_scaled)[0]
        confidence = round(max(probability) * 100, 2)

        if prediction[0] == 1:
            st.markdown('<div class="result-box-danger">⚠️ HIGH RISK: The person is DIABETIC</div>', unsafe_allow_html=True)
            st.error(f"🎯 Confidence: {confidence}%")
            risk = "High" if confidence > 80 else "Medium"
            st.markdown(f'<div class="tip-box">🚨 Risk Level: <b>{risk}</b><br><br>💡 Health Tips:<br>• Monitor blood sugar daily<br>• Reduce sugar intake<br>• Exercise 30 mins daily<br>• Consult a doctor immediately</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box-success">✅ LOW RISK: The person is NOT Diabetic</div>', unsafe_allow_html=True)
            st.success(f"🎯 Confidence: {confidence}%")
            st.markdown('<div class="tip-box">✅ Risk Level: <b>Low</b><br><br>💡 Stay Healthy Tips:<br>• Maintain healthy diet<br>• Exercise regularly<br>• Regular health checkups<br>• Stay hydrated</div>', unsafe_allow_html=True)

# =================== HEART ===================
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    st.markdown("Enter patient details below to predict heart disease risk.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
        trestbps = st.number_input("Resting Blood Pressure", 0, 250)
        chol = st.number_input("Cholesterol", 0, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.number_input("Resting ECG (0-2)", 0, 2)
    with col2:
        thalach = st.number_input("Max Heart Rate", 0, 250)
        exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0)
        slope = st.number_input("Slope (0-2)", 0, 2)
        ca = st.number_input("Major Vessels (0-4)", 0, 4)
        thal = st.number_input("Thal (0-3)", 0, 3)

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = heart_scaler.transform(input_data)
        prediction = heart_model.predict(input_scaled)
        probability = heart_model.predict_proba(input_scaled)[0]
        confidence = round(max(probability) * 100, 2)

        if prediction[0] == 1:
            st.markdown('<div class="result-box-danger">⚠️ HIGH RISK: The person HAS Heart Disease</div>', unsafe_allow_html=True)
            st.error(f"🎯 Confidence: {confidence}%")
            risk = "High" if confidence > 80 else "Medium"
            st.markdown(f'<div class="tip-box">🚨 Risk Level: <b>{risk}</b><br><br>💡 Health Tips:<br>• Avoid oily and fatty food<br>• Exercise daily (light walking)<br>• Monitor blood pressure<br>• Consult cardiologist immediately</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box-success">✅ LOW RISK: The person does NOT have Heart Disease</div>', unsafe_allow_html=True)
            st.success(f"🎯 Confidence: {confidence}%")
            st.markdown('<div class="tip-box">✅ Risk Level: <b>Low</b><br><br>💡 Stay Healthy Tips:<br>• Eat heart healthy foods<br>• Regular cardio exercise<br>• Avoid smoking and alcohol<br>• Annual heart checkup</div>', unsafe_allow_html=True)

# =================== PARKINSONS ===================
elif disease == "Parkinsons":
    st.header("🧠 Parkinsons Prediction")
    st.markdown("Enter patient voice measurement details below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0)
        MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 600.0)
        MDVP_Flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0)
        MDVP_Jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0)
        MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 1.0)
        MDVP_RAP = st.number_input("MDVP:RAP", 0.0, 1.0)
        MDVP_PPQ = st.number_input("MDVP:PPQ", 0.0, 1.0)
        Jitter_DDP = st.number_input("Jitter:DDP", 0.0, 1.0)
    with col2:
        MDVP_Shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0)
        MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", 0.0, 5.0)
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3", 0.0, 1.0)
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5", 0.0, 1.0)
        MDVP_APQ = st.number_input("MDVP:APQ", 0.0, 1.0)
        Shimmer_DDA = st.number_input("Shimmer:DDA", 0.0, 1.0)
        NHR = st.number_input("NHR", 0.0, 1.0)
        HNR = st.number_input("HNR", 0.0, 50.0)
    with col3:
        RPDE = st.number_input("RPDE", 0.0, 1.0)
        DFA = st.number_input("DFA", 0.0, 1.0)
        spread1 = st.number_input("Spread1", -10.0, 0.0)
        spread2 = st.number_input("Spread2", 0.0, 1.0)
        D2 = st.number_input("D2", 0.0, 5.0)
        PPE = st.number_input("PPE", 0.0, 1.0)

    if st.button("🔍 Predict Parkinsons", use_container_width=True):
        input_data = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,
                                MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                                MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
                                Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                                RPDE, DFA, spread1, spread2, D2, PPE]])
        input_scaled = parkinsons_scaler.transform(input_data)
        prediction = parkinsons_model.predict(input_scaled)
        probability = parkinsons_model.predict_proba(input_scaled)[0]
        confidence = round(max(probability) * 100, 2)

        if prediction[0] == 1:
            st.markdown('<div class="result-box-danger">⚠️ HIGH RISK: The person HAS Parkinsons</div>', unsafe_allow_html=True)
            st.error(f"🎯 Confidence: {confidence}%")
            risk = "High" if confidence > 80 else "Medium"
            st.markdown(f'<div class="tip-box">🚨 Risk Level: <b>{risk}</b><br><br>💡 Health Tips:<br>• Consult neurologist immediately<br>• Physical therapy recommended<br>• Medication management<br>• Support group assistance</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box-success">✅ LOW RISK: The person does NOT have Parkinsons</div>', unsafe_allow_html=True)
            st.success(f"🎯 Confidence: {confidence}%")
            st.markdown('<div class="tip-box">✅ Risk Level: <b>Low</b><br><br>💡 Stay Healthy Tips:<br>• Regular brain exercises<br>• Stay physically active<br>• Healthy diet with antioxidants<br>• Regular neurological checkup</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#888;">Made with ❤️ | Multiple Disease Prediction System | ML Project</p>', unsafe_allow_html=True)