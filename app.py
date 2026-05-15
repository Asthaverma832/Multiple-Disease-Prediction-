import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import requests

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
.chat-box {
    background: rgba(255,255,255,0.05);
    padding: 15px; border-radius: 10px; margin: 10px 0; color: white;
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

# PDF Generator
def generate_pdf(disease, result, confidence, tips):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(200, 750, "Multiple Disease Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Disease: {disease}")
    c.drawString(50, 700, f"Result: {result}")
    c.drawString(50, 680, f"Confidence: {confidence}%")
    c.drawString(50, 660, "Health Tips:")
    y = 640
    for tip in tips:
        c.drawString(70, y, f"• {tip}")
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

# Gauge chart
def create_gauge(confidence, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': title, 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': "#00d2ff"},
            'steps': [
                {'range': [0, 40], 'color': '#38ef7d'},
                {'range': [40, 70], 'color': '#ffd700'},
                {'range': [70, 100], 'color': '#ff416c'}
            ],
        },
        number={'suffix': '%', 'font': {'color': 'white'}}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Pie chart
def create_pie(positive, negative):
    fig = go.Figure(go.Pie(
        labels=['Disease Probability', 'Healthy Probability'],
        values=[positive, negative],
        marker=dict(colors=['#ff416c', '#38ef7d']),
        hole=0.4
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# Bar chart
def create_bar_chart(values, labels, normal_values, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Patient', x=labels, y=values, marker_color='#00d2ff'))
    fig.add_trace(go.Bar(name='Normal', x=labels, y=normal_values, marker_color='#38ef7d'))
    fig.update_layout(title=title, barmode='group',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,0.8)',
                      font=dict(color='white'), height=350,
                      legend=dict(font=dict(color='white')))
    return fig

# AI Doctor Chatbot
def get_ai_response(question, disease_context):
    health_tips = {
        "diabetes": {
            "diet": "Avoid sugar, eat whole grains, vegetables and lean proteins.",
            "exercise": "Walk 30 minutes daily, do light cardio 5 days a week.",
            "medicine": "Consult doctor for Metformin or insulin therapy.",
            "symptoms": "Frequent urination, excessive thirst, blurred vision are common symptoms.",
            "default": "Monitor blood sugar daily, maintain healthy weight, avoid smoking."
        },
        "heart": {
            "diet": "Eat heart healthy foods - omega 3, fruits, vegetables. Avoid oily food.",
            "exercise": "Light walking, yoga, avoid intense exercise without doctor advice.",
            "medicine": "Consult cardiologist for proper medication.",
            "symptoms": "Chest pain, shortness of breath, irregular heartbeat are warning signs.",
            "default": "Monitor blood pressure daily, avoid stress, quit smoking."
        },
        "parkinsons": {
            "diet": "Mediterranean diet with antioxidants helps. Avoid constipating foods.",
            "exercise": "Physical therapy, balance exercises, tai chi recommended.",
            "medicine": "Levodopa is common medication - consult neurologist.",
            "symptoms": "Tremors, stiffness, slow movement, balance problems.",
            "default": "Regular neurologist visits, physical therapy, support group."
        }
    }

    question_lower = question.lower()
    tips = health_tips.get(disease_context.lower(), health_tips["diabetes"])

    if any(word in question_lower for word in ["diet", "food", "eat", "khana"]):
        return tips["diet"]
    elif any(word in question_lower for word in ["exercise", "workout", "vyayam"]):
        return tips["exercise"]
    elif any(word in question_lower for word in ["medicine", "drug", "tablet", "dawai"]):
        return tips["medicine"]
    elif any(word in question_lower for word in ["symptom", "sign", "lakshan"]):
        return tips["symptoms"]
    else:
        return tips["default"]

# Title
st.markdown('<p class="title-text">🏥 Multiple Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Powered Early Disease Detection Using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
st.sidebar.title("Navigation")
disease = st.sidebar.selectbox("🔍 Select Disease", 
    ["Diabetes", "Heart Disease", "Parkinsons", "BMI Calculator", "AI Doctor Chat"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Accuracy")
st.sidebar.success("🩸 Diabetes: 72.08%")
st.sidebar.warning("❤️ Heart Disease: 98.54%")
st.sidebar.info("🧠 Parkinsons: 89.74%")

# =================== BMI CALCULATOR ===================
if disease == "BMI Calculator":
    st.header("⚖️ BMI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0)
    with col2:
        height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)

    if st.button("Calculate BMI", use_container_width=True):
        bmi = round(weight / ((height/100) ** 2), 2)
        if bmi < 18.5:
            category, color, tip = "Underweight", "blue", "Eat more nutritious food."
        elif bmi < 25:
            category, color, tip = "Normal Weight", "green", "Great! Maintain your lifestyle."
        elif bmi < 30:
            category, color, tip = "Overweight", "orange", "Exercise regularly."
        else:
            category, color, tip = "Obese", "red", "Consult a doctor immediately."

        st.markdown(f"### Your BMI: **{bmi}** — :{color}[{category}]")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=bmi,
            title={'text': "BMI Meter", 'font': {'color': 'white'}},
            gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#00d2ff"},
                   'steps': [{'range': [0, 18.5], 'color': '#4169e1'},
                              {'range': [18.5, 25], 'color': '#38ef7d'},
                              {'range': [25, 30], 'color': '#ffd700'},
                              {'range': [30, 50], 'color': '#ff416c'}]},
            number={'font': {'color': 'white'}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="tip-box">💡 {tip}</div>', unsafe_allow_html=True)

# =================== AI DOCTOR CHAT ===================
elif disease == "AI Doctor Chat":
    st.header("🤖 AI Doctor Chatbot")
    st.markdown("Ask health questions related to Diabetes, Heart Disease, or Parkinsons!")

    disease_topic = st.selectbox("Select Disease Topic", ["Diabetes", "Heart", "Parkinsons"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your health question:", placeholder="e.g. What diet should I follow?")

    if st.button("Ask AI Doctor 🤖", use_container_width=True):
        if user_input:
            response = get_ai_response(user_input, disease_topic)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI Doctor", response))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f'<div class="chat-box">🧑 <b>You:</b> {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-box">🤖 <b>AI Doctor:</b> {message}</div>', unsafe_allow_html=True)

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# =================== DIABETES ===================
elif disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
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

        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                result = "DIABETIC"
                tips = ["Monitor blood sugar daily", "Reduce sugar intake",
                        "Exercise 30 mins daily", "Consult doctor immediately"]
                st.markdown('<div class="result-box-danger">⚠️ DIABETIC</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">🚨 Risk: <b>High</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)
            else:
                result = "NOT Diabetic"
                tips = ["Maintain healthy diet", "Exercise regularly", "Regular checkups"]
                st.markdown('<div class="result-box-success">✅ NOT Diabetic</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">✅ Risk: <b>Low</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)

            pdf = generate_pdf("Diabetes", result, confidence, tips)
            st.download_button("📄 Download Report PDF", pdf, "diabetes_report.pdf", "application/pdf")

        with col2:
            st.plotly_chart(create_gauge(confidence, "Confidence Score"), use_container_width=True)
            st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

        patient_vals = [Glucose, BloodPressure, BMI, Age]
        normal_vals = [100, 80, 22, 30]
        labels = ['Glucose', 'Blood Pressure', 'BMI', 'Age']
        st.plotly_chart(create_bar_chart(patient_vals, labels, normal_vals, "Patient vs Normal Values"),
                        use_container_width=True)

# =================== HEART ===================
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
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

        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                result = "Heart Disease Detected"
                tips = ["Avoid oily food", "Light walking daily",
                        "Monitor blood pressure", "Consult cardiologist"]
                st.markdown('<div class="result-box-danger">⚠️ Heart Disease Detected</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">🚨 Risk: <b>High</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)
            else:
                result = "No Heart Disease"
                tips = ["Eat heart healthy foods", "Regular cardio", "Avoid smoking"]
                st.markdown('<div class="result-box-success">✅ No Heart Disease</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">✅ Risk: <b>Low</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)

            pdf = generate_pdf("Heart Disease", result, confidence, tips)
            st.download_button("📄 Download Report PDF", pdf, "heart_report.pdf", "application/pdf")

        with col2:
            st.plotly_chart(create_gauge(confidence, "Confidence Score"), use_container_width=True)
            st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

        patient_vals = [trestbps, chol, thalach, age]
        normal_vals = [120, 200, 150, 40]
        labels = ['Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'Age']
        st.plotly_chart(create_bar_chart(patient_vals, labels, normal_vals, "Patient vs Normal Values"),
                        use_container_width=True)

# =================== PARKINSONS ===================
elif disease == "Parkinsons":
    st.header("🧠 Parkinsons Prediction")
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

        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                result = "Parkinsons Detected"
                tips = ["Consult neurologist", "Physical therapy", "Medication management"]
                st.markdown('<div class="result-box-danger">⚠️ Parkinsons Detected</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">🚨 Risk: <b>High</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)
            else:
                result = "No Parkinsons"
                tips = ["Regular brain exercises", "Stay physically active", "Healthy diet"]
                st.markdown('<div class="result-box-success">✅ No Parkinsons</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tip-box">✅ Risk: <b>Low</b><br>{"<br>".join("• "+t for t in tips)}</div>',
                            unsafe_allow_html=True)

            pdf = generate_pdf("Parkinsons", result, confidence, tips)
            st.download_button("📄 Download Report PDF", pdf, "parkinsons_report.pdf", "application/pdf")

        with col2:
            st.plotly_chart(create_gauge(confidence, "Confidence Score"), use_container_width=True)
            st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#888;">Made with ❤️ | Multiple Disease Prediction System | ML Project</p>',
            unsafe_allow_html=True)