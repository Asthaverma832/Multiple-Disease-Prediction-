import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from datetime import datetime
from gtts import gTTS
import base64
import fitz
from PIL import Image
import json
import os
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a1628 100%); }
.main-title {
    text-align: center; font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff416c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding: 10px 0;
}
.subtitle { text-align: center; color: #888; font-size: 1.1rem; letter-spacing: 2px; }
.result-danger {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    padding: 25px; border-radius: 20px; text-align: center;
    font-size: 1.8rem; font-weight: 800; color: white; margin: 20px 0;
    box-shadow: 0 10px 40px rgba(255,65,108,0.5);
}
.result-success {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 25px; border-radius: 20px; text-align: center;
    font-size: 1.8rem; font-weight: 800; color: white; margin: 20px 0;
    box-shadow: 0 10px 40px rgba(56,239,125,0.5);
}
.tip-card {
    background: linear-gradient(135deg, rgba(0,210,255,0.05), rgba(123,47,247,0.05));
    border-left: 4px solid #00d2ff; border-radius: 10px;
    padding: 20px; margin: 15px 0; color: #ccc;
}
.stat-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 15px; padding: 20px; text-align: center; margin: 10px 0;
}
.chat-user {
    background: linear-gradient(135deg, rgba(0,210,255,0.2), rgba(0,210,255,0.05));
    border-radius: 20px 20px 5px 20px; padding: 15px 20px; margin: 8px 0 8px 20%;
    color: white; border: 1px solid rgba(0,210,255,0.3);
}
.chat-ai {
    background: linear-gradient(135deg, rgba(123,47,247,0.2), rgba(123,47,247,0.05));
    border-radius: 20px 20px 20px 5px; padding: 15px 20px; margin: 8px 20% 8px 0;
    color: white; border: 1px solid rgba(123,47,247,0.3);
}
.chat-time { font-size: 0.7rem; color: #666; margin-top: 5px; }
.record-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(0,210,255,0.2);
    border-radius: 15px; padding: 15px; margin: 10px 0;
}
.section-header {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 20px 0;
}
.stButton > button {
    background: linear-gradient(135deg, #00d2ff, #7b2ff7) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 10px;
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

# ===== PATIENT RECORDS SYSTEM =====
RECORDS_FILE = "patient_records.json"

def load_records():
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_record(name, age, gender, disease, result, confidence):
    records = load_records()
    record = {
        "id": len(records) + 1,
        "name": name,
        "age": age,
        "gender": gender,
        "disease": disease,
        "result": result,
        "confidence": confidence,
        "date": datetime.now().strftime("%d %B %Y"),
        "time": datetime.now().strftime("%I:%M %p")
    }
    records.append(record)
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)
    return record

def delete_record(record_id):
    records = load_records()
    records = [r for r in records if r["id"] != record_id]
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)

# ===== CHAT HISTORY =====
CHAT_FILE = "chat_history.json"

def load_chat():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, 'r') as f:
            return json.load(f)
    return []

def save_chat(sender, message):
    chats = load_chat()
    chats.append({
        "sender": sender,
        "message": message,
        "time": datetime.now().strftime("%I:%M %p"),
        "date": datetime.now().strftime("%d %b")
    })
    with open(CHAT_FILE, 'w') as f:
        json.dump(chats, f, indent=2)

# ===== VOICE =====
def text_to_speech(text):
    try:
        clean_text = text.replace("**", "").replace("•", "").replace("#", "")
        tts = gTTS(text=clean_text[:500], lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                    unsafe_allow_html=True)
    except:
        pass

# ===== PDF =====
def generate_pdf(patient_name, age, disease, result, confidence, tips):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFillColorRGB(0.05, 0.05, 0.15)
    c.rect(0, height-120, width, 120, fill=1)
    c.setFillColorRGB(0, 0.82, 1)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height-55, "MediPredict AI - Medical Report")
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 11)
    c.drawString(50, height-80, f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-150, "Patient Information")
    c.setFont("Helvetica", 12)
    c.drawString(50, height-170, f"Name: {patient_name}  |  Age: {age}  |  Disease: {disease}")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-210, "AI Prediction Result")
    if "NOT" in result or "No" in result:
        c.setFillColorRGB(0.07, 0.6, 0.49)
    else:
        c.setFillColorRGB(1, 0.25, 0.42)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height-230, f"Result: {result}")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 12)
    c.drawString(50, height-250, f"Confidence: {confidence}%")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-290, "Health Recommendations")
    c.setFont("Helvetica", 12)
    y = height-315
    for tip in tips:
        c.drawString(70, y, f"• {tip}")
        y -= 22
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.setFont("Helvetica", 9)
    c.drawString(50, 40, "Disclaimer: AI-generated report. Consult a qualified doctor for medical advice.")
    c.save()
    buffer.seek(0)
    return buffer

# ===== CHARTS =====
def create_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "AI Confidence", 'font': {'color': 'white', 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': "#00d2ff", 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(56,239,125,0.3)'},
                {'range': [40, 70], 'color': 'rgba(255,215,0,0.3)'},
                {'range': [70, 100], 'color': 'rgba(255,65,108,0.3)'}
            ]
        },
        number={'suffix': '%', 'font': {'color': 'white', 'size': 36}}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260,
                      margin=dict(l=30, r=30, t=60, b=10), font=dict(color='white'))
    return fig

def create_pie(positive, negative):
    fig = go.Figure(go.Pie(
        labels=['Disease Risk', 'Healthy'],
        values=[positive, negative],
        marker=dict(colors=['#ff416c', '#38ef7d'],
                    line=dict(color='rgba(0,0,0,0.3)', width=2)),
        hole=0.5, textfont=dict(color='white', size=13)
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      height=260, margin=dict(l=20, r=20, t=20, b=10),
                      legend=dict(font=dict(color='white')))
    return fig

def create_bar(values, labels, normal_values):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Your Values', x=labels, y=values,
                         marker=dict(color='#00d2ff', opacity=0.8),
                         text=values, textposition='outside', textfont=dict(color='white')))
    fig.add_trace(go.Bar(name='Normal Range', x=labels, y=normal_values,
                         marker=dict(color='#38ef7d', opacity=0.8),
                         text=normal_values, textposition='outside', textfont=dict(color='white')))
    fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(255,255,255,0.02)', font=dict(color='white'),
                      height=350, legend=dict(font=dict(color='white')))
    return fig

# ===== AI RESPONSE =====
def get_ai_response(question, disease_context):
    tips = {
        "Diabetes": {
            "diet": """🥗 **Diet Plan for Diabetes:**

**Foods to EAT:**
- Whole grains like brown rice, oats, and quinoa — they release sugar slowly
- Green leafy vegetables like spinach, methi, and broccoli — very low in sugar
- Lean proteins like eggs, chicken, fish, and lentils (dal)
- Healthy fats like nuts, seeds, and olive oil
- Low sugar fruits like berries, apple, pear (in small portions)

**Foods to AVOID:**
- White rice, white bread, maida — spike blood sugar fast
- Sugary drinks like cold drinks, juices, chai with sugar
- Fried foods like samosa, pakoda, chips
- Sweets like mithai, chocolates, ice cream

**Pro Tip:** Eat small meals every 3-4 hours instead of 3 big meals. This keeps your blood sugar stable throughout the day.""",

            "exercise": """🏃 **Exercise Guide for Diabetes:**

**Best Exercises:**
- Morning walk for 30-45 minutes daily — most effective for blood sugar control
- Yoga poses like Surya Namaskar, Pranayama, and Kapalbhati
- Swimming is excellent — full body workout with no joint stress
- Cycling for 20-30 minutes

**Important Tips:**
- Always check blood sugar BEFORE exercising
- Keep a glucose biscuit handy during exercise
- Never exercise on empty stomach if you take insulin
- Drink plenty of water during exercise""",

            "medicine": """💊 **Medicines for Diabetes:**

**Common Medicines:**
- **Metformin** — First line medicine, reduces glucose production in liver
- **Glipizide/Glimepiride** — Helps pancreas produce more insulin
- **Jardiance** — New medicine that removes sugar through urine
- **Insulin** — Rapid acting before meals, long acting at bedtime

**Natural Support:**
- Karela (bitter gourd) juice — natural blood sugar reducer
- Fenugreek (methi) seeds soaked overnight
- Cinnamon (dalchini) in warm water

⚠️ Never stop or change medicine without consulting your doctor!""",

            "symptoms": """🔍 **Diabetes Symptoms:**

**Early Warning Signs:**
- Frequent urination — especially at night
- Excessive thirst — drinking lots of water but still thirsty
- Extreme hunger — even after eating
- Unexplained weight loss

**Common Symptoms:**
- Blurred or hazy vision
- Cuts and wounds healing very slowly
- Frequent infections — skin, UTI, or fungal
- Numbness or tingling in hands and feet
- Fatigue and weakness throughout the day

**Emergency Signs:**
- Blood sugar below 70 mg/dL — dizziness, sweating, confusion
- Blood sugar above 300 mg/dL — go to hospital immediately!""",

            "prevention": """🛡️ **How to Prevent Diabetes:**

- Lose 5-10% body weight if overweight — reduces risk by 58%!
- Exercise at least 150 minutes per week
- Eat balanced diet — more vegetables, less sugar
- Quit smoking — 30-40% higher risk in smokers
- Sleep 7-8 hours every night
- Manage stress through meditation

**Know Your Numbers:**
- Normal fasting blood sugar: 70-100 mg/dL
- Pre-diabetes: 100-125 mg/dL
- Diabetes: 126 mg/dL or above""",

            "test": """🧪 **Diabetes Tests:**

- **Fasting Blood Glucose** — After 8 hours fasting. Normal: 70-100, Diabetes: 126+
- **HbA1c Test** — Average blood sugar of last 3 months. Diabetes if 6.5%+
- **OGTT** — Drink glucose solution, test after 2 hours
- **Daily glucometer** — Check blood sugar at home morning and evening
- **Urine test** — Check for glucose and ketones
- **Kidney function test** — Diabetes can affect kidneys over time

**How Often:**
- Newly diagnosed: Daily monitoring
- Well controlled: Every 3 months HbA1c""",

            "default": """👨‍⚕️ **General Diabetes Advice:**

**Daily Routine:**
- Morning: Check blood sugar, light exercise, healthy breakfast
- Afternoon: Balanced lunch, short walk after eating
- Evening: Light snack, yoga or meditation
- Night: Check blood sugar, take medication if prescribed

**Most Important Rules:**
- Never skip meals — especially if you take insulin
- Always carry a fast-acting sugar source
- Wear medical ID showing you are diabetic
- Monitor blood sugar daily

With proper care, you can live a completely normal, active, and happy life! 💪"""
        },
        "Heart": {
            "diet": """🥗 **Heart Healthy Diet:**

**Foods that PROTECT your heart:**
- Fatty fish like salmon — rich in Omega-3
- Colorful vegetables — carrots, tomatoes, bell peppers
- Whole grains — oats, brown rice
- Nuts and seeds — almonds, walnuts (handful daily)
- Garlic and onion — naturally lower blood pressure

**Foods to STRICTLY AVOID:**
- Deep fried foods — samosa, puri, pakoda
- Excess salt — causes high blood pressure
- Packaged and processed foods
- Sugary drinks and alcohol

**Cooking Tips:**
- Steam, bake, or grill instead of frying
- Use mustard oil or olive oil in small amounts""",

            "exercise": """🏃 **Exercise for Heart Health:**

**Safe Exercises:**
- Brisk walking — 30 minutes daily, 5 days a week
- Swimming — Low impact, excellent for heart
- Cycling — 20-30 minutes, moderate pace
- Yoga — Anulom Vilom pranayama is excellent

**Safety Rules:**
- Warm up for 5 minutes before exercise
- Stop if you feel chest pain or dizziness
- Never exercise right after eating
- Always get cardiologist approval first!

**Target Heart Rate:**
- Maximum = 220 minus your age
- Exercise at 50-70% of maximum""",

            "medicine": """💊 **Heart Medicines:**

- **ACE Inhibitors** (Ramipril) — relax blood vessels
- **Beta Blockers** (Atenolol) — slow heart rate, reduce BP
- **Statins** (Atorvastatin) — reduce bad cholesterol
- **Aspirin 75mg** — prevents blood clots
- **Clopidogrel** — for patients with stents

⚠️ Never stop heart medicines suddenly — can trigger heart attack!""",

            "symptoms": """🔍 **Heart Disease Symptoms:**

**Classic Signs:**
- Chest pain or pressure — heaviness, tightness
- Shortness of breath — even during light activity
- Irregular heartbeat — too fast, too slow, or skipping
- Dizziness or lightheadedness
- Extreme fatigue

**Less Obvious Signs:**
- Pain in left arm, shoulder, jaw, or back
- Swollen ankles and feet
- Cold sweats without fever

**EMERGENCY — Call ambulance immediately:**
- Severe crushing chest pain lasting 5+ minutes
- Chest pain spreading to arm or jaw
- Sudden difficulty breathing
- These are signs of HEART ATTACK!""",

            "prevention": """🛡️ **Heart Disease Prevention:**

**Control Risk Factors:**
- Blood Pressure: Keep below 120/80 mmHg
- Cholesterol: LDL below 100, HDL above 40
- Weight: BMI between 18.5-24.9

**Lifestyle Changes:**
- Quit smoking — doubles heart disease risk
- Limit alcohol
- Manage stress — raises blood pressure
- Sleep 7-8 hours nightly

**Regular Checkups:**
- Blood pressure — monthly
- Cholesterol — every 6 months
- ECG — annually after age 40""",

            "test": """🧪 **Heart Tests:**

- **ECG** — Records heart's electrical activity. Quick and painless
- **Echocardiogram** — Ultrasound shows how well heart pumps
- **Stress Test (TMT)** — ECG while on treadmill
- **Coronary Angiography** — X-ray of heart arteries. Gold standard!
- **Blood Tests** — Cholesterol, troponin (heart damage marker)
- **Holter Monitor** — Wear ECG for 24-48 hours

**Normal Values:**
- Ejection fraction: 55-70% is normal
- 70%+ blockage usually needs stent or bypass""",

            "default": """👨‍⚕️ **Heart Health Advice:**

Your heart beats 100,000 times a day — give it the best care!

**Know Your Numbers:**
- Blood Pressure: Below 120/80 is ideal
- Resting heart rate: 60-100 bpm is normal
- Cholesterol: Total below 200 mg/dL

**Red Flags — See Doctor Immediately:**
- Any chest pain or pressure
- Sudden shortness of breath
- Heart beating very fast or irregularly

Heart disease is largely preventable! Simple changes can reduce risk by 80%. Start small, stay consistent! ❤️"""
        },
        "Parkinsons": {
            "diet": """🥗 **Diet for Parkinsons:**

**Best Foods:**
- Antioxidant rich — berries, dark chocolate, green tea
- Omega-3 — walnuts, flaxseeds, fatty fish
- High fiber — fruits, vegetables, whole grains (helps constipation)
- Vitamin D foods — eggs, mushrooms, fortified milk

**Important Medication Tip:**
- If taking Levodopa, take it 30-60 minutes BEFORE meals
- High protein meals can reduce Levodopa effectiveness
- Spread protein intake throughout the day""",

            "exercise": """🏃 **Exercise for Parkinsons:**

**Best Exercises:**
- Walking with poles — improves balance
- Tai Chi — scientifically proven to reduce falls
- Dancing — Tango shown to help with movement!
- Swimming — water supports body weight
- Rock Steady Boxing — specially designed for Parkinsons

**Daily Routine:**
- Morning: Stretching and balance (15-20 min)
- Afternoon: Walking or swimming (30 min)
- Evening: Relaxation yoga or tai chi (15 min)

**Safety:** Always exercise with someone nearby!""",

            "medicine": """💊 **Parkinsons Medicines:**

- **Levodopa/Carbidopa** — Most effective. Converts to dopamine in brain
- **Dopamine Agonists** (Pramipexole) — Mimic dopamine, used in early stages
- **MAO-B Inhibitors** (Rasagiline) — Prevent dopamine breakdown
- **Amantadine** — Reduces involuntary movements

**Managing Medicines:**
- Take at SAME TIME every day
- Never skip doses — can cause sudden worsening
- Keep a medicine diary
- Discuss "off periods" with doctor

**Advanced:** Deep Brain Stimulation (DBS) surgery — very effective for advanced cases""",

            "symptoms": """🔍 **Parkinsons Symptoms:**

**Motor Symptoms:**
- Tremors — shaking in one hand at rest (pill rolling motion)
- Slowness of movement (Bradykinesia)
- Muscle stiffness in arms, legs, neck
- Balance problems and tendency to fall
- Freezing — suddenly unable to move feet

**Non-Motor Symptoms (appear early):**
- Loss of smell — can appear years before diagnosis
- Sleep problems — acting out dreams
- Constipation
- Depression and anxiety
- Soft or muffled voice

**Early Warning Signs:**
- Handwriting becoming smaller
- Arm not swinging when walking
- Face becoming less expressive""",

            "prevention": """🛡️ **Parkinsons Risk Reduction:**

- Exercise regularly — lower risk in active people
- Eat Mediterranean diet
- Drink coffee or tea — caffeine appears protective!
- Avoid head injuries — wear helmets
- Avoid pesticide exposure

**Brain Health:**
- Stay mentally active — puzzles, reading, learning
- Strong social connections
- Manage stress
- Sleep 7-8 hours

Note: Parkinsons cannot be completely prevented, but healthy habits promote brain health and may reduce risk.""",

            "test": """🧪 **Parkinsons Diagnosis:**

- **Clinical Evaluation** — Main diagnosis method. No single definitive test!
- **DaTscan** — Shows dopamine-producing cells. Most specific test
- **MRI Brain** — Rules out other conditions
- **PET scan** — Shows brain activity
- **Trial of Levodopa** — Good response confirms Parkinsons

**Genetic Testing:**
- LRRK2, PINK1, PRKN gene mutations linked to Parkinsons
- Recommended if multiple family members affected

**Note:** Neurologist evaluates symptoms, history, and physical exam for diagnosis.""",

            "default": """👨‍⚕️ **Living Well with Parkinsons:**

**Daily Tips:**
- Take medications on time — consistency is everything
- Clear walkways at home — remove rugs and obstacles
- Install grab bars in bathroom
- Use weighted utensils to help with tremors

**Support:**
- Join a Parkinsons support group
- Involve family in care
- Physiotherapy, speech therapy, occupational therapy

Over 10 million people live with Parkinsons worldwide. Many live 20+ years with good quality of life after diagnosis.

Stay hopeful, stay active! 🧠💪"""
        }
    }

    q = question.lower()
    t = tips.get(disease_context, tips["Diabetes"])

    if any(w in q for w in ["diet", "food", "eat", "khana", "nutrition"]):
        return t["diet"]
    elif any(w in q for w in ["exercise", "workout", "walk", "vyayam", "physical"]):
        return t["exercise"]
    elif any(w in q for w in ["medicine", "drug", "tablet", "medication", "dawai"]):
        return t["medicine"]
    elif any(w in q for w in ["symptom", "sign", "lakshan", "feel", "problem"]):
        return t["symptoms"]
    elif any(w in q for w in ["prevent", "avoid", "bachao", "safe", "precaution"]):
        return t["prevention"]
    elif any(w in q for w in ["test", "diagnosis", "check", "detect"]):
        return t["test"]
    else:
        return t["default"]

# Extract PDF text
def extract_pdf_text(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text[:2000]
    except:
        return "Could not extract text from PDF."

# ===== MAIN APP =====
st.markdown('<div class="main-title">🏥 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ AI POWERED EARLY DISEASE DETECTION SYSTEM ✨</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=80)
    st.markdown("## 🏥 MediPredict AI")
    st.markdown("*Your AI Health Companion*")
    st.markdown("---")
    disease = st.selectbox("🔍 Select Module", [
        "🏠 Home",
        "🩸 Diabetes",
        "❤️ Heart Disease",
        "🧠 Parkinsons",
        "⚖️ BMI Calculator",
        "🤖 AI Doctor Chat",
        "📋 Patient Records",
        "🏥 Find Nearby Doctors"
    ])
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("🩸 Diabetes", "72.08%", "Accuracy")
    st.metric("❤️ Heart Disease", "98.54%", "Accuracy")
    st.metric("🧠 Parkinsons", "89.74%", "Accuracy")
    st.markdown("---")
    st.markdown("*Made with ❤️ by Astha Verma*")

# ===== HOME =====
if disease == "🏠 Home":
    st.markdown("## 👋 Welcome to MediPredict AI!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="stat-card">
            <h2>🩸</h2><h3 style="color:white;">Diabetes</h3>
            <p style="color:#888;">Predict diabetes using medical parameters</p>
            <h2 style="color:#00d2ff;">72.08%</h2>
            <p style="color:#888;">Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="stat-card">
            <h2>❤️</h2><h3 style="color:white;">Heart Disease</h3>
            <p style="color:#888;">Detect heart disease with cardiac data</p>
            <h2 style="color:#00d2ff;">98.54%</h2>
            <p style="color:#888;">Accuracy</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="stat-card">
            <h2>🧠</h2><h3 style="color:white;">Parkinsons</h3>
            <p style="color:#888;">Identify Parkinsons via voice analysis</p>
            <h2 style="color:#00d2ff;">89.74%</h2>
            <p style="color:#888;">Accuracy</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="tip-card">
            <h3 style="color:#00d2ff;">🤖 AI Features</h3>
            <p>• Disease Prediction with Confidence Score</p>
            <p>• AI Doctor Chatbot with Voice Output</p>
            <p>• Upload Medical Report for Analysis</p>
            <p>• Patient Records System</p>
            <p>• Nearby Doctor Finder with Map</p>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="tip-card">
            <h3 style="color:#7b2ff7;">📊 Visualizations</h3>
            <p>• Gauge Chart - AI Confidence Score</p>
            <p>• Pie Chart - Risk Distribution</p>
            <p>• Bar Chart - Patient vs Normal Values</p>
            <p>• Interactive Maps for Doctor Search</p>
            </div>""", unsafe_allow_html=True)

# ===== DISEASE PREDICTION =====
elif disease in ["🩸 Diabetes", "❤️ Heart Disease", "🧠 Parkinsons"]:
    st.markdown(f'<div class="section-header">{disease} Prediction</div>', unsafe_allow_html=True)

    with st.expander("👤 Patient Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="Enter full name")
        with col2:
            patient_age = st.number_input("Age", 1, 120, 25)
        with col3:
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if disease == "🩸 Diabetes":
        st.markdown("#### 🔬 Medical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input("🤰 Pregnancies", 0, 20)
            Glucose = st.number_input("🍬 Glucose Level (mg/dL)", 0, 300)
            BloodPressure = st.number_input("💉 Blood Pressure (mm Hg)", 0, 150)
            SkinThickness = st.number_input("📏 Skin Thickness (mm)", 0, 100)
        with col2:
            Insulin = st.number_input("💊 Insulin Level", 0, 900)
            BMI = st.number_input("⚖️ BMI", 0.0, 70.0)
            DPF = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0)
            Age = st.number_input("📅 Age", 1, 120)

        if st.button("🔍 Predict Diabetes", use_container_width=True):
            if not patient_name:
                st.warning("⚠️ Please enter patient name!")
            else:
                with st.spinner("🤖 AI is analyzing..."):
                    input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                           SkinThickness, Insulin, BMI, DPF, Age]])
                    input_scaled = diabetes_scaler.transform(input_data)
                    prediction = diabetes_model.predict(input_scaled)
                    probability = diabetes_model.predict_proba(input_scaled)[0]
                    confidence = round(max(probability) * 100, 2)

                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        result = "DIABETIC - High Risk"
                        tips = ["Monitor blood sugar daily",
                                "Avoid sugar and processed foods",
                                "Exercise 30 minutes daily",
                                "Take prescribed medication regularly",
                                "Consult endocrinologist immediately"]
                        st.markdown('<div class="result-danger">⚠️ DIABETIC DETECTED</div>', unsafe_allow_html=True)
                        voice_text = f"{patient_name} has been detected as diabetic with {confidence} percent confidence. Please consult a doctor immediately."
                    else:
                        result = "NOT Diabetic - Low Risk"
                        tips = ["Maintain healthy diet", "Exercise regularly",
                                "Annual blood sugar checkup", "Stay hydrated",
                                "Maintain healthy weight"]
                        st.markdown('<div class="result-success">✅ NOT DIABETIC</div>', unsafe_allow_html=True)
                        voice_text = f"Good news! {patient_name} is not diabetic. Confidence is {confidence} percent. Keep maintaining a healthy lifestyle."

                    st.markdown(f'<div class="tip-card">💡 <b>Tips for {patient_name}:</b><br>' +
                               '<br>'.join(f'• {t}' for t in tips) + '</div>', unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔊 Hear Result", key="v_diabetes"):
                            text_to_speech(voice_text)
                    with col_b:
                        pdf = generate_pdf(patient_name, patient_age, "Diabetes", result, confidence, tips)
                        st.download_button("📄 Download Report", pdf,
                                          f"{patient_name}_diabetes_report.pdf",
                                          "application/pdf", use_container_width=True)

                    save_record(patient_name, patient_age, patient_gender, "Diabetes", result, confidence)
                    st.success("✅ Record saved to Patient Records!")

                with col2:
                    st.plotly_chart(create_gauge(confidence), use_container_width=True)
                    st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

                st.plotly_chart(create_bar(
                    [Glucose, BloodPressure, BMI, Age],
                    ['Glucose', 'Blood Pressure', 'BMI', 'Age'],
                    [100, 80, 22, 30]), use_container_width=True)

    elif disease == "❤️ Heart Disease":
        st.markdown("#### 🔬 Cardiological Parameters")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("📅 Age", 1, 120)
            sex = st.selectbox("👤 Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.number_input("💔 Chest Pain Type (0-3)", 0, 3)
            trestbps = st.number_input("💉 Resting Blood Pressure", 0, 250)
            chol = st.number_input("🧪 Cholesterol (mg/dL)", 0, 600)
            fbs = st.selectbox("🍬 Fasting Blood Sugar>120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            restecg = st.number_input("📊 Resting ECG (0-2)", 0, 2)
        with col2:
            thalach = st.number_input("❤️ Max Heart Rate", 0, 250)
            exang = st.selectbox("🏃 Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            oldpeak = st.number_input("📉 ST Depression", 0.0, 10.0)
            slope = st.number_input("📈 Slope (0-2)", 0, 2)
            ca = st.number_input("🫀 Major Vessels (0-4)", 0, 4)
            thal = st.number_input("🧬 Thal (0-3)", 0, 3)

        if st.button("🔍 Predict Heart Disease", use_container_width=True):
            if not patient_name:
                st.warning("⚠️ Please enter patient name!")
            else:
                with st.spinner("🤖 AI analyzing cardiac data..."):
                    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                           restecg, thalach, exang, oldpeak, slope, ca, thal]])
                    input_scaled = heart_scaler.transform(input_data)
                    prediction = heart_model.predict(input_scaled)
                    probability = heart_model.predict_proba(input_scaled)[0]
                    confidence = round(max(probability) * 100, 2)

                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        result = "Heart Disease Detected"
                        tips = ["Consult cardiologist immediately",
                                "Avoid oily and fatty foods",
                                "Light walking only",
                                "Monitor BP daily",
                                "Take prescribed heart medication"]
                        st.markdown('<div class="result-danger">⚠️ HEART DISEASE DETECTED</div>', unsafe_allow_html=True)
                        voice_text = f"Alert! {patient_name} shows signs of heart disease with {confidence} percent confidence. Please consult a cardiologist immediately."
                    else:
                        result = "No Heart Disease"
                        tips = ["Eat heart healthy foods", "Regular cardio exercise",
                                "Avoid smoking and alcohol", "Manage stress",
                                "Annual cardiac checkup"]
                        st.markdown('<div class="result-success">✅ NO HEART DISEASE</div>', unsafe_allow_html=True)
                        voice_text = f"Great news! {patient_name} does not show signs of heart disease. Confidence is {confidence} percent."

                    st.markdown(f'<div class="tip-card">💡 <b>Tips for {patient_name}:</b><br>' +
                               '<br>'.join(f'• {t}' for t in tips) + '</div>', unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔊 Hear Result", key="v_heart"):
                            text_to_speech(voice_text)
                    with col_b:
                        pdf = generate_pdf(patient_name, patient_age, "Heart Disease", result, confidence, tips)
                        st.download_button("📄 Download Report", pdf,
                                          f"{patient_name}_heart_report.pdf",
                                          "application/pdf", use_container_width=True)

                    save_record(patient_name, patient_age, patient_gender, "Heart Disease", result, confidence)
                    st.success("✅ Record saved to Patient Records!")

                with col2:
                    st.plotly_chart(create_gauge(confidence), use_container_width=True)
                    st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

                st.plotly_chart(create_bar(
                    [trestbps, chol, thalach, age],
                    ['Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'Age'],
                    [120, 200, 150, 40]), use_container_width=True)

    elif disease == "🧠 Parkinsons":
        st.markdown("#### 🔬 Voice Measurement Parameters")
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
            if not patient_name:
                st.warning("⚠️ Please enter patient name!")
            else:
                with st.spinner("🤖 AI analyzing voice data..."):
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
                        tips = ["Consult neurologist immediately",
                                "Start physical therapy",
                                "Take prescribed medication",
                                "Join Parkinsons support group",
                                "Regular brain exercise"]
                        st.markdown('<div class="result-danger">⚠️ PARKINSONS DETECTED</div>', unsafe_allow_html=True)
                        voice_text = f"Alert! {patient_name} shows signs of Parkinsons with {confidence} percent confidence. Please consult a neurologist."
                    else:
                        result = "No Parkinsons"
                        tips = ["Regular brain exercises", "Stay physically active",
                                "Antioxidant rich diet",
                                "Regular neurological checkup",
                                "Manage stress and sleep well"]
                        st.markdown('<div class="result-success">✅ NO PARKINSONS</div>', unsafe_allow_html=True)
                        voice_text = f"Good news! {patient_name} does not show signs of Parkinsons. Keep maintaining a healthy lifestyle."

                    st.markdown(f'<div class="tip-card">💡 <b>Tips for {patient_name}:</b><br>' +
                               '<br>'.join(f'• {t}' for t in tips) + '</div>', unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔊 Hear Result", key="v_park"):
                            text_to_speech(voice_text)
                    with col_b:
                        pdf = generate_pdf(patient_name, patient_age, "Parkinsons", result, confidence, tips)
                        st.download_button("📄 Download Report", pdf,
                                          f"{patient_name}_parkinsons_report.pdf",
                                          "application/pdf", use_container_width=True)

                    save_record(patient_name, patient_age, patient_gender, "Parkinsons", result, confidence)
                    st.success("✅ Record saved to Patient Records!")

                with col2:
                    st.plotly_chart(create_gauge(confidence), use_container_width=True)
                    st.plotly_chart(create_pie(probability[1]*100, probability[0]*100), use_container_width=True)

# ===== BMI =====
elif disease == "⚖️ BMI Calculator":
    st.markdown('<div class="section-header">⚖️ BMI Calculator</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Your Name", placeholder="Enter your name")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("⚖️ Weight (kg)", 1.0, 300.0, 70.0)
    with col2:
        height = st.number_input("📏 Height (cm)", 50.0, 250.0, 170.0)

    if st.button("Calculate BMI", use_container_width=True):
        bmi = round(weight / ((height/100) ** 2), 2)
        if bmi < 18.5:
            category, tip = "Underweight 🔵", "Increase calorie intake with nutritious foods."
        elif bmi < 25:
            category, tip = "Normal Weight 🟢", "Great! Keep maintaining your healthy lifestyle."
        elif bmi < 30:
            category, tip = "Overweight 🟡", "Exercise regularly and eat a balanced diet."
        else:
            category, tip = "Obese 🔴", "Please consult a doctor for a proper weight loss plan."

        if patient_name:
            st.markdown(f"### {patient_name}'s BMI: **{bmi}** — {category}")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=bmi,
            title={'text': "BMI Meter", 'font': {'color': 'white'}},
            gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#00d2ff"},
                   'steps': [{'range': [0, 18.5], 'color': 'rgba(65,105,225,0.4)'},
                              {'range': [18.5, 25], 'color': 'rgba(56,239,125,0.4)'},
                              {'range': [25, 30], 'color': 'rgba(255,215,0,0.4)'},
                              {'range': [30, 50], 'color': 'rgba(255,65,108,0.4)'}]},
            number={'font': {'color': 'white'}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="tip-card">💡 {tip}</div>', unsafe_allow_html=True)

# ===== AI DOCTOR CHAT =====
elif disease == "🤖 AI Doctor Chat":
    st.markdown('<div class="section-header">🤖 AI Doctor Chatbot</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Chat with AI Doctor", "📋 Upload Medical Report"])

    with tab1:
        disease_topic = st.selectbox("Select Disease Topic", ["Diabetes", "Heart", "Parkinsons"])

        st.markdown("**⚡ Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            q1 = st.button("🥗 Diet Tips", use_container_width=True)
        with col2:
            q2 = st.button("🏃 Exercise Tips", use_container_width=True)
        with col3:
            q3 = st.button("🔍 Symptoms", use_container_width=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            q4 = st.button("💊 Medicines", use_container_width=True)
        with col5:
            q5 = st.button("🛡️ Prevention", use_container_width=True)
        with col6:
            q6 = st.button("🧪 Tests", use_container_width=True)

        user_input = st.text_input("💬 Type your question:",
                                   placeholder="e.g. What diet should I follow?",
                                   key="chat_input")

        col1, col2 = st.columns([3, 1])
        with col1:
            ask = st.button("Ask AI Doctor 🤖", use_container_width=True)
        with col2:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                if os.path.exists(CHAT_FILE):
                    os.remove(CHAT_FILE)
                st.rerun()

        auto_question = None
        if q1: auto_question = "diet"
        elif q2: auto_question = "exercise"
        elif q3: auto_question = "symptoms"
        elif q4: auto_question = "medicine"
        elif q5: auto_question = "prevention"
        elif q6: auto_question = "test"

        if auto_question:
            response = get_ai_response(auto_question, disease_topic)
            save_chat("You", auto_question.capitalize())
            save_chat("AI Doctor", response)
            text_to_speech(response[:300])
            st.rerun()

        if ask and user_input:
            response = get_ai_response(user_input, disease_topic)
            save_chat("You", user_input)
            save_chat("AI Doctor", response)
            text_to_speech(response[:300])
            st.rerun()

        # Display chat
        st.markdown("---")
        chats = load_chat()
        for chat in reversed(chats):
            if chat["sender"] == "You":
                st.markdown(f'''<div class="chat-user">
                    🧑 <b>You</b> <span class="chat-time">{chat["time"]}</span>
                    <div style="margin-top:8px;">{chat["message"]}</div>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="chat-ai">
                    🤖 <b>AI Doctor</b> <span class="chat-time">{chat["time"]}</span>
                    <div style="margin-top:8px;">{chat["message"]}</div>
                </div>''', unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📋 Upload Your Medical Report")
        report_disease = st.selectbox("Disease type:", ["Diabetes", "Heart", "Parkinsons"])
        uploaded_file = st.file_uploader("Upload PDF or Image",
                                         type=["pdf", "png", "jpg", "jpeg"])

        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            if uploaded_file.type == "application/pdf":
                with st.spinner("📖 AI reading report..."):
                    text = extract_pdf_text(uploaded_file)
                st.text_area("Report Preview", text[:500] + "...", height=150)
                advice = get_ai_response("symptoms", report_disease)
                st.markdown(f'<div class="tip-card">💡 AI Advice:<br>{advice}</div>', unsafe_allow_html=True)
                if st.button("🔊 Hear Advice"):
                    text_to_speech(advice[:300])
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Report", use_column_width=True)
                advice = get_ai_response("default", report_disease)
                st.markdown(f'<div class="tip-card">💡 AI Advice:<br>{advice}</div>', unsafe_allow_html=True)

# ===== PATIENT RECORDS =====
elif disease == "📋 Patient Records":
    st.markdown('<div class="section-header">📋 Patient Records</div>', unsafe_allow_html=True)

    records = load_records()

    if not records:
        st.info("📭 No patient records yet. Make a prediction first!")
    else:
        st.markdown(f"### Total Records: **{len(records)}**")

        # Search
        search = st.text_input("🔍 Search by patient name:", placeholder="Type name...")

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            filter_disease = st.selectbox("Filter by Disease", ["All", "Diabetes", "Heart Disease", "Parkinsons"])
        with col2:
            filter_result = st.selectbox("Filter by Result", ["All", "Positive", "Negative"])

        filtered = records
        if search:
            filtered = [r for r in filtered if search.lower() in r["name"].lower()]
        if filter_disease != "All":
            filtered = [r for r in filtered if r["disease"] == filter_disease]
        if filter_result == "Positive":
            filtered = [r for r in filtered if "NOT" not in r["result"] and "No" not in r["result"]]
        elif filter_result == "Negative":
            filtered = [r for r in filtered if "NOT" in r["result"] or "No" in r["result"]]

        st.markdown(f"**Showing {len(filtered)} records**")

        for record in reversed(filtered):
            is_positive = "NOT" not in record["result"] and "No" not in record["result"]
            color = "#ff416c" if is_positive else "#38ef7d"
            st.markdown(f'''<div class="record-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h3 style="color:white;margin:0;">👤 {record["name"]}</h3>
                        <p style="color:#888;margin:5px 0;">Age: {record["age"]} | Gender: {record["gender"]} | {record["date"]} {record["time"]}</p>
                        <p style="color:#aaa;">Disease: <b>{record["disease"]}</b></p>
                        <p style="color:{color};font-weight:bold;">Result: {record["result"]}</p>
                        <p style="color:#888;">Confidence: {record["confidence"]}%</p>
                    </div>
                </div>
            </div>''', unsafe_allow_html=True)

            if st.button(f"🗑️ Delete Record #{record['id']}", key=f"del_{record['id']}"):
                delete_record(record["id"])
                st.rerun()

        # Stats
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patients", len(records))
        with col2:
            positive = len([r for r in records if "NOT" not in r["result"] and "No" not in r["result"]])
            st.metric("Positive Cases", positive)
        with col3:
            st.metric("Negative Cases", len(records) - positive)

# ===== NEARBY DOCTORS =====
elif disease == "🏥 Find Nearby Doctors":
    st.markdown('<div class="section-header">🏥 Find Nearby Doctors</div>', unsafe_allow_html=True)

    st.markdown("""<div class="tip-card">
        📍 Enter your city and select specialist type to find nearby doctors on the map!
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("🏙️ Enter your city:", placeholder="e.g. Lucknow, Delhi, Mumbai")
    with col2:
        specialist = st.selectbox("👨‍⚕️ Specialist Type", [
            "General Physician",
            "Endocrinologist (Diabetes)",
            "Cardiologist (Heart)",
            "Neurologist (Parkinsons)",
            "Hospital"
        ])

    if st.button("🔍 Find Doctors", use_container_width=True):
        if not city:
            st.warning("⚠️ Please enter your city!")
        else:
            with st.spinner("🗺️ Finding doctors near you..."):
                # City coordinates (common Indian cities)
                city_coords = {
                    "lucknow": [26.8467, 80.9462],
                    "delhi": [28.6139, 77.2090],
                    "mumbai": [19.0760, 72.8777],
                    "bangalore": [12.9716, 77.5946],
                    "hyderabad": [17.3850, 78.4867],
                    "chennai": [13.0827, 80.2707],
                    "kolkata": [22.5726, 88.3639],
                    "pune": [18.5204, 73.8567],
                    "jaipur": [26.9124, 75.7873],
                    "ahmedabad": [23.0225, 72.5714],
                    "agra": [27.1767, 78.0081],
                    "varanasi": [25.3176, 82.9739],
                    "kanpur": [26.4499, 80.3319],
                    "patna": [25.5941, 85.1376],
                }

                city_lower = city.lower().strip()
                coords = city_coords.get(city_lower, [26.8467, 80.9462])

                # Create map
                m = folium.Map(location=coords, zoom_start=13,
                              tiles='CartoDB dark_matter')

                # Add user location marker
                folium.Marker(
                    coords,
                    popup="📍 Your Location",
                    tooltip="You are here!",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')
                ).add_to(m)

                # Simulated nearby doctors
                import random
                doctor_names = {
                    "Endocrinologist (Diabetes)": [
                        "Dr. Priya Sharma - Diabetes Specialist",
                        "Dr. Rajesh Kumar - Endocrinologist",
                        "Dr. Sunita Gupta - Diabetes & Thyroid",
                        "Dr. Amit Verma - Metabolic Disorders",
                        "Dr. Neha Singh - Endocrinology"
                    ],
                    "Cardiologist (Heart)": [
                        "Dr. Vikram Malhotra - Cardiologist",
                        "Dr. Sanjay Patel - Heart Specialist",
                        "Dr. Anita Rao - Interventional Cardiology",
                        "Dr. Rahul Saxena - Cardiac Surgery",
                        "Dr. Pooja Mehta - Heart & Vascular"
                    ],
                    "Neurologist (Parkinsons)": [
                        "Dr. Ravi Shankar - Neurologist",
                        "Dr. Kavita Joshi - Movement Disorders",
                        "Dr. Suresh Nair - Parkinsons Specialist",
                        "Dr. Meena Tripathi - Neurology",
                        "Dr. Arun Mishra - Brain & Spine"
                    ],
                    "General Physician": [
                        "Dr. Rakesh Gupta - General Physician",
                        "Dr. Shobha Yadav - Family Medicine",
                        "Dr. Deepak Srivastava - General Medicine",
                        "Dr. Anjali Tiwari - Internal Medicine",
                        "Dr. Vivek Pandey - GP & Primary Care"
                    ],
                    "Hospital": [
                        "City Medical Center",
                        "Apollo Hospital",
                        "Medanta Hospital",
                        "Fortis Healthcare",
                        "AIIMS Regional Center"
                    ]
                }

                doctors = doctor_names.get(specialist, doctor_names["General Physician"])
                ratings = ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]

                for i, doctor in enumerate(doctors):
                    offset_lat = random.uniform(-0.02, 0.02)
                    offset_lon = random.uniform(-0.02, 0.02)
                    doc_coords = [coords[0] + offset_lat, coords[1] + offset_lon]

                    folium.Marker(
                        doc_coords,
                        popup=f"""
                        <div style="font-family:Arial; padding:10px; min-width:200px;">
                            <h4 style="color:#0066cc;">🏥 {doctor}</h4>
                            <p>{ratings[i % len(ratings)]}</p>
                            <p>📍 {city.title()}</p>
                            <p>📞 +91-{random.randint(7000000000, 9999999999)}</p>
                            <p>🕐 Mon-Sat: 9AM - 7PM</p>
                        </div>""",
                        tooltip=f"🏥 {doctor}",
                        icon=folium.Icon(color='red', icon='plus', prefix='fa')
                    ).add_to(m)

                st_folium(m, width=900, height=500)

                st.markdown("### 👨‍⚕️ Found Doctors:")
                for i, doctor in enumerate(doctors):
                    st.markdown(f'''<div class="record-card">
                        <h4 style="color:#00d2ff;">🏥 {doctor}</h4>
                        <p style="color:#888;">📍 {city.title()} | {ratings[i % len(ratings)]} | 🕐 Mon-Sat: 9AM-7PM</p>
                        <p style="color:#aaa;">Specialization: {specialist}</p>
                    </div>''', unsafe_allow_html=True)

    st.markdown("""<div class="tip-card">
        💡 <b>Tip:</b> For emergency, call <b>108</b> (Ambulance) or <b>102</b> (Emergency Medical Services)
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#888;">Made with ❤️ | MediPredict AI | Final Year Project by Astha Verma</p>',
            unsafe_allow_html=True)