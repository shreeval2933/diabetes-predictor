import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PIMA INDIANS DIABETES RISK DECISION-MAKING USING BAYESIAN NETWORK",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - DARK MODE ONLY
st.markdown("""
    <style>
    /* Dark Mode Background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Header - White text for dark background */
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 8px rgba(102, 126, 234, 0.5);
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.5);
    }
    .low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .medium-risk {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Info box - Dark theme */
    .info-box {
        background: rgba(30, 58, 138, 0.3);
        padding: 15px;
        border-left: 4px solid #3b82f6;
        border-radius: 5px;
        margin: 10px 0;
        color: #e0f2fe;
    }
    .info-box h4 {
        color: #60a5fa;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.5);
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .status-ready {
        background: #10b981;
        color: white;
    }
    
    /* Dark mode text colors */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0e7ff !important;
    }
    
    p, div, span, label {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a5b4fc !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    .streamlit-expanderHeader {
        background-color: rgba(30, 58, 138, 0.3) !important;
        color: #e0e7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pretrained_model(model_path='models/bn_model.pkl'):
    """Load pre-trained Bayesian Network model"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please run `python train_and_save_model.py` to create the model.")
            return None, None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        structure = model_data.get('structure', None)
        
        return model, structure
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def discretize_input(data):
    """Discretize continuous input values according to model training"""
    discrete_data = {}
    
    # Age
    if data['Age'] <= 30:
        discrete_data['Age'] = 'Young'
    elif data['Age'] <= 40:
        discrete_data['Age'] = 'Middle'
    elif data['Age'] <= 50:
        discrete_data['Age'] = 'Senior'
    else:
        discrete_data['Age'] = 'Elderly'
    
    # Pregnancies
    if data['Pregnancies'] == 0:
        discrete_data['Pregnancies'] = 'None'
    elif data['Pregnancies'] <= 3:
        discrete_data['Pregnancies'] = 'Low'
    elif data['Pregnancies'] <= 6:
        discrete_data['Pregnancies'] = 'Medium'
    else:
        discrete_data['Pregnancies'] = 'High'
    
    # Glucose
    if data['Glucose'] <= 100:
        discrete_data['Glucose'] = 'Normal'
    elif data['Glucose'] <= 125:
        discrete_data['Glucose'] = 'Prediabetes'
    else:
        discrete_data['Glucose'] = 'Diabetes'
    
    # Blood Pressure
    if data['BloodPressure'] <= 80:
        discrete_data['BloodPressure'] = 'Normal'
    elif data['BloodPressure'] <= 90:
        discrete_data['BloodPressure'] = 'Elevated'
    else:
        discrete_data['BloodPressure'] = 'High'
    
    # BMI
    if data['BMI'] < 18.5:
        discrete_data['BMI'] = 'Underweight'
    elif data['BMI'] < 25:
        discrete_data['BMI'] = 'Normal'
    elif data['BMI'] < 30:
        discrete_data['BMI'] = 'Overweight'
    else:
        discrete_data['BMI'] = 'Obese'
    
    # Skin Thickness
    if data['SkinThickness'] <= 20:
        discrete_data['SkinThickness'] = 'Low'
    elif data['SkinThickness'] <= 30:
        discrete_data['SkinThickness'] = 'Medium'
    else:
        discrete_data['SkinThickness'] = 'High'
    
    # Insulin
    if data['Insulin'] <= 100:
        discrete_data['Insulin'] = 'Low'
    elif data['Insulin'] <= 200:
        discrete_data['Insulin'] = 'Medium'
    else:
        discrete_data['Insulin'] = 'High'
    
    # Diabetes Pedigree Function
    if data['DiabetesPedigreeFunction'] <= 0.3:
        discrete_data['DiabetesPedigreeFunction'] = 'Low'
    elif data['DiabetesPedigreeFunction'] <= 0.6:
        discrete_data['DiabetesPedigreeFunction'] = 'Medium'
    else:
        discrete_data['DiabetesPedigreeFunction'] = 'High'
    
    return discrete_data

def predict_diabetes(model, evidence):
    """Make prediction using the trained Bayesian Network"""
    try:
        inference = VariableElimination(model)
        
        result = inference.query(
            variables=['Outcome'],
            evidence=evidence,
            show_progress=False
        )
        
        # Get probability of diabetes (Outcome='1')
        if len(result.values) > 1:
            probability = result.values[1]
        else:
            probability = result.values[0]
        
        return probability
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5

def create_gauge_chart(probability):
    """Create a gauge chart for risk visualization - Dark theme"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 24, 'color': '#e0e7ff', 'family': 'Arial Black'}},
        number={'suffix': "%", 'font': {'size': 50, 'color': '#ffffff', 'family': 'Arial Black'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#ef4444" if probability > 0.5 else "#3b82f6", 'thickness': 0.8},
            'bgcolor': "#1e293b",
            'borderwidth': 3,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 30], 'color': '#064e3b'},
                {'range': [30, 70], 'color': '#78350f'},
                {'range': [70, 100], 'color': '#7f1d1d'}
            ],
            'threshold': {
                'line': {'color': "#a78bfa", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#0f0f1e",
        plot_bgcolor="#0f0f1e",
        font={'color': "#e0e7ff", 'family': "Arial"},
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_feature_bars(data):
    """Create horizontal bar chart for input features - Dark theme"""
    features = list(data.keys())
    values = list(data.values())
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', 
              '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
    
    fig = go.Figure(go.Bar(
        y=features,
        x=[1] * len(features),
        orientation='h',
        text=values,
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        marker=dict(
            color=colors,
            line=dict(color='#1e293b', width=2)
        ),
        hovertemplate='<b>%{y}</b><br>Value: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Your Health Profile",
        title_font=dict(size=20, color='#e0e7ff', family='Arial Black'),
        showlegend=False,
        height=500,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=14, color='#cbd5e1', family='Arial')),
        paper_bgcolor='#0f0f1e',
        plot_bgcolor='#0f0f1e',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="big-font">🏥 PIMA INDIANS DIABETES RISK DECISION-MAKING USING BAYESIAN NETWORK </p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 18px; margin-top: -20px;">Powered by Pre-trained Bayesian Network AI</p>', unsafe_allow_html=True)
    
    # Load pre-trained model
    with st.spinner("🔄 Loading pre-trained model..."):
        model, structure = load_pretrained_model('models/bn_model.pkl')
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure 'models/bn_model.pkl' exists.")
        st.info("💡 Run `python train_and_save_model.py` locally to create the model, then add it to your repository.")
        st.stop()
    
    # Success message
    st.success("✅ Pre-trained Bayesian Network model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
        st.title("📋 Patient Information")
        st.markdown("---")
        
        # Model status
        st.markdown('<div class="status-badge status-ready">🟢 Model Ready</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("Enter Patient Data")
        
        # Input fields with better styling and help text
        pregnancies = st.slider(
            "👶 Number of Pregnancies", 
            0, 17, 3,
            help="Total number of pregnancies"
        )
        
        glucose = st.slider(
            "🍬 Glucose Level (mg/dL)", 
            0, 200, 120,
            help="Plasma glucose concentration (Normal: <100, Prediabetes: 100-125, Diabetes: >125)"
        )
        
        blood_pressure = st.slider(
            "💓 Blood Pressure (mm Hg)", 
            0, 122, 70,
            help="Diastolic blood pressure (Normal: <80, Elevated: 80-90, High: >90)"
        )
        
        skin_thickness = st.slider(
            "📏 Skin Thickness (mm)", 
            0, 99, 20,
            help="Triceps skin fold thickness"
        )
        
        insulin = st.slider(
            "💉 Insulin Level (μU/mL)", 
            0, 846, 79,
            help="2-Hour serum insulin level"
        )
        
        bmi = st.slider(
            "⚖️ BMI", 
            0.0, 67.1, 31.4, 0.1,
            help="Body Mass Index (Normal: 18.5-25, Overweight: 25-30, Obese: >30)"
        )
        
        dpf = st.slider(
            "🧬 Diabetes Pedigree Function", 
            0.078, 2.42, 0.47, 0.001,
            help="Genetic influence factor (family history)"
        )
        
        age = st.slider(
            "👤 Age (years)", 
            21, 81, 33,
            help="Age in years"
        )
        
        st.markdown("---")
        predict_button = st.button("🔍 Predict Risk", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Input Summary")
        
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # Display in a nice card format
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("👶 Pregnancies", pregnancies)
            st.metric("💓 Blood Pressure", f"{blood_pressure} mm Hg")
            st.metric("💉 Insulin", f"{insulin} μU/mL")
            st.metric("🧬 Pedigree", f"{dpf:.3f}")
        
        with metric_col2:
            st.metric("🍬 Glucose", f"{glucose} mg/dL")
            st.metric("📏 Skin Thickness", f"{skin_thickness} mm")
            st.metric("⚖️ BMI", f"{bmi:.1f}")
            st.metric("👤 Age", f"{age} years")
    
    with col2:
        st.markdown("### 🎯 Health Profile Visualization")
        discrete_data = discretize_input(input_data)
        fig_bars = create_feature_bars(discrete_data)
        st.plotly_chart(fig_bars, use_container_width=True)
    
    # Prediction section
    if predict_button:
        with st.spinner('🔬 Analyzing your health data with Bayesian Network...'):
            # Discretize input
            evidence = discretize_input(input_data)
            
            # Make prediction
            probability = predict_diabetes(model, evidence)
            
            st.markdown("---")
            st.markdown("## 📈 Prediction Results")
            
            # Result columns
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            
            with res_col2:
                # Gauge chart
                fig_gauge = create_gauge_chart(probability)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk classification with detailed message
                if probability < 0.3:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        ✅ LOW RISK<br>
                        <span style="font-size: 18px;">Probability: {probability*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✨ Your diabetes risk is low. Keep maintaining a healthy lifestyle!")
                    st.info("💡 Continue regular exercise, balanced diet, and routine check-ups.")
                    
                elif probability < 0.7:
                    st.markdown(f"""
                    <div class="prediction-box medium-risk">
                        ⚠️ MODERATE RISK<br>
                        <span style="font-size: 18px;">Probability: {probability*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ You have moderate risk. Consider preventive measures.")
                    st.info("💡 Consult with a healthcare provider for a comprehensive evaluation and preventive strategies.")
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        🚨 HIGH RISK<br>
                        <span style="font-size: 18px;">Probability: {probability*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("🚨 High diabetes risk detected. Medical consultation recommended.")
                    st.info("💡 Please schedule an appointment with a healthcare provider as soon as possible for proper evaluation and management.")
            
            # Show discretized evidence
            with st.expander("🔍 View Discretized Input (Evidence Used for Prediction)"):
                evidence_df = pd.DataFrame([evidence]).T
                evidence_df.columns = ['Category']
                st.dataframe(evidence_df, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("""
                <div class="info-box">
                    <h4>🥗 Lifestyle Modifications</h4>
                    <ul>
                        <li><b>Diet:</b> Low glycemic index foods, reduce sugar intake</li>
                        <li><b>Exercise:</b> 150 minutes/week moderate activity</li>
                        <li><b>Weight:</b> Maintain healthy BMI (18.5-24.9)</li>
                        <li><b>Monitoring:</b> Regular blood glucose checks</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("""
                <div class="info-box">
                    <h4>🏥 Medical Follow-up</h4>
                    <ul>
                        <li><b>Testing:</b> HbA1c test every 3-6 months</li>
                        <li><b>Consultation:</b> Endocrinologist/Nutritionist</li>
                        <li><b>Screening:</b> Annual comprehensive health check</li>
                        <li><b>Management:</b> Blood pressure and cholesterol control</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factor analysis
            st.markdown("---")
            st.markdown("### 📊 Key Risk Factors in Your Profile")
            
            risk_factors = []
            protective_factors = []
            
            # Analyze each factor
            if discrete_data['Glucose'] in ['Prediabetes', 'Diabetes']:
                risk_factors.append(f"🔴 **Glucose Level**: {discrete_data['Glucose']} - Major risk factor")
            else:
                protective_factors.append(f"🟢 **Glucose Level**: {discrete_data['Glucose']} - Within normal range")
            
            if discrete_data['BMI'] in ['Overweight', 'Obese']:
                risk_factors.append(f"🔴 **BMI**: {discrete_data['BMI']} ({bmi:.1f}) - Weight management needed")
            elif discrete_data['BMI'] == 'Normal':
                protective_factors.append(f"🟢 **BMI**: {discrete_data['BMI']} ({bmi:.1f}) - Healthy weight")
            
            if discrete_data['Age'] in ['Senior', 'Elderly']:
                risk_factors.append(f"🔴 **Age**: {discrete_data['Age']} ({age} years) - Age is a risk factor")
            else:
                protective_factors.append(f"🟢 **Age**: {discrete_data['Age']} ({age} years) - Younger age group")
            
            if discrete_data['DiabetesPedigreeFunction'] in ['Medium', 'High']:
                risk_factors.append(f"🔴 **Family History**: {discrete_data['DiabetesPedigreeFunction']} genetic influence")
            else:
                protective_factors.append(f"🟢 **Family History**: {discrete_data['DiabetesPedigreeFunction']} genetic influence")
            
            if discrete_data['BloodPressure'] in ['Elevated', 'High']:
                risk_factors.append(f"🔴 **Blood Pressure**: {discrete_data['BloodPressure']} ({blood_pressure} mm Hg) - Needs attention")
            else:
                protective_factors.append(f"🟢 **Blood Pressure**: {discrete_data['BloodPressure']} ({blood_pressure} mm Hg) - Normal")
            
            # Display factors
            factor_col1, factor_col2 = st.columns(2)
            
            with factor_col1:
                st.markdown("#### ⚠️ Risk Factors")
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("No significant risk factors identified!")
            
            with factor_col2:
                st.markdown("#### ✅ Protective Factors")
                if protective_factors:
                    for factor in protective_factors:
                        st.markdown(factor)
                else:
                    st.info("Consider improving protective factors")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 20px;'>
        <p><strong>⚠️ Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.</p>
        <p style='margin-top: 10px;'>© 2024 Diabetes Risk Predictor | Powered by Bayesian Network AI with Structure Learning</p>
        <p style='font-size: 12px; margin-top: 10px;'>Model: Hill Climb Search + BIC Scoring | Estimation: Bayesian (BDeu Prior)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
