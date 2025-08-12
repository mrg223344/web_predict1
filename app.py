import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
# Use st.set_page_config() as the first Streamlit command.
st.set_page_config(
    page_title="Cognitive Frailty Risk Assessment",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads the SVM model from the specified path."""
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Please run the script to create the dummy model first.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model('svm_model.pkl')

# --- Mappings and Dictionaries (Centralized Configuration) ---

# Feature names (English to Chinese)
feature_map = {
    'Age': '年龄',
    'educational level': '文化程度',
    'Regular exercise': '规律运动',
    'DCCN': '糖尿病慢性并发症数量',
    'Malnutrition': '营养不良',
    'Depressive': '抑郁'
}

# Reverse mapping for processing (Chinese to English)
reverse_feature_map = {v: k for k, v in feature_map.items()}

# Feature options with their numerical encodings
feature_options = {
    '年龄': {'60-69岁': 0, '70-79岁': 1, '80岁及以上': 2},
    '文化程度': {'小学及以下': 0, '初中': 1, '高中/中专/技校': 2, '大专及以上': 3},
    '规律运动': {'否': 0, '是': 1},
    '糖尿病慢性并发症数量': {'<2个': 0, '≥2个': 1},
    '营养不良': {'否': 0, '是': 1},
    '抑郁': {'否': 0, '是': 1}
}

# --- Translations for Multilingual Support ---
translations = {
    'zh-CN': {
        "app_title": "老年2型糖尿病患者认知衰弱风险评估",
        "form_title": "老年2型糖尿病患者认知衰弱风险评估",
        "result_title": "评估结果",
        "language": "English",
        "submit_btn": "开始评估",
        "back_btn": "返回首页",
        "select_placeholder": "- 请选择",
        "input_summary_title": "您提供的信息",
        "risk_probability_label": "存在认知衰弱的概率",
        "result_explanation": "此概率为模型通过支持向量机算法计算得出，表示您的特征组合与认知衰弱患者的匹配程度。该结果仅供参考，不能替代专业医学诊断。",
        "risk_level_low": "低风险",
        "risk_level_moderate": "中风险",
        "risk_level_high": "高风险",
        # Feature Names
        '年龄': '年龄', '文化程度': '文化程度', '规律运动': '规律运动',
        '糖尿病慢性并发症数量': '糖尿病慢性并发症数量', '营养不良': '营养不良', '抑郁': '抑郁',
        # Feature Options
        '60-69岁': '60-69岁', '70-79岁': '70-79岁', '80岁及以上': '80岁及以上',
        '小学及以下': '小学及以下', '初中': '初中', '高中/中专/技校': '高中/中专/技校', '大专及以上': '大专及以上',
        '否': '否', '是': '是', '<2个': '<2个', '≥2个': '≥2个'
    },
    'en': {
        "app_title": "Cognitive Frailty Risk Assessment",
        "form_title": "Cognitive Frailty Risk Assessment for Elderly Patients with Type 2 Diabetes",
        "result_title": "Assessment Result",
        "language": "中文",
        "submit_btn": "Start Assessment",
        "back_btn": "Back to Home",
        "select_placeholder": "- Select",
        "input_summary_title": "Your Input Information",
        "risk_probability_label": "Probability of Cognitive Frailty",
        "result_explanation": "This probability is calculated by the SVM algorithm, representing the match between your features and cognitive frailty cases. For reference only, not a substitute for professional medical diagnosis.",
        "risk_level_low": "Low Risk",
        "risk_level_moderate": "Moderate Risk",
        "risk_level_high": "High Risk",
        # Feature Names
        '年龄': 'Age', '文化程度': 'Educational Level', '规律运动': 'Regular Exercise',
        '糖尿病慢性并发症数量': 'Number of Chronic Diabetic Complications', '营养不良': 'Malnutrition', '抑郁': 'Depression',
        # Feature Options
        '60-69岁': '60-69 years', '70-79岁': '70-79 years', '80岁及以上': '80+ years',
        '小学及以下': 'Primary school or below', '初中': 'Junior high school',
        '高中/中专/技校': 'High school/Technical school', '大专及以上': 'College or above',
        '否': 'No', '是': 'Yes', '<2个': '<2', '≥2个': '≥2'
    }
}

# --- Helper function for translation ---
def t(key):
    """Retrieves a translated string for the given key."""
    return translations[st.session_state.current_lang].get(key, key)

# --- CSS Styling (copied from your Flask templates) ---
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary: #2563eb;
            --success: #059669;
            --warning: #eab308;
            --danger: #dc2626;
            --bg-light: #f8fafc;
            --bg-card: #ffffff;
            --text-dark: #1a365d;
            --text-light: #4b5563;
        }
        /* General styling */
        .stApp {
            background-color: var(--bg-light);
        }
        /* Hide Streamlit's default header and footer */
        .st-emotion-cache-18ni7ap, .st-emotion-cache-h4xjwg {
            display: none;
        }
        /* Main container styling */
        .main .block-container {
            max-width: 800px;
            width: 100%;
            border-radius: 20px;
            background: var(--bg-card);
            box-shadow: 0 10px 30px rgba(34, 41, 51, 0.08);
            padding: 40px;
        }
        /* Header styling */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
        }
        .app-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-dark);
        }
        .app-logo .icon {
            font-size: 1.7rem;
            color: var(--primary);
        }
        /* Custom Button for Language */
        .stButton>button {
            background: var(--primary);
            color: white;
            padding: 8px 16px;
            border-radius: 12px;
            border: none;
            font-size: 0.9rem;
            transition: transform 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            background: #1d4ed8;
            color: white;
        }
        /* Result card styling */
        .result-card {
            background: var(--bg-light);
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            margin: 40px 0;
        }
        .risk-value {
            font-size: 4rem;
            font-weight: 700;
            color: var(--primary);
            margin: 20px 0 10px;
        }
        .risk-label {
            font-size: 1.1rem;
            color: var(--text-light);
            margin-bottom: 20px;
        }
        .explanation {
            max-width: 600px;
            margin: 0 auto;
            font-size: 0.95rem;
            line-height: 1.7;
            color: var(--text-light);
        }
        /* Input summary styling */
        .input-summary {
            border-top: 1px solid #e5e7eb;
            padding-top: 30px;
        }
        .summary-title {
            text-align: center;
            color: var(--text-dark);
            font-size: 1.2rem;
            margin-bottom: 25px;
        }
        .input-item {
            background: var(--bg-light);
            padding: 18px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.95rem;
            color: var(--text-light);
            margin-bottom: 12px;
        }
        /* Adding Remixicon CDN */
        <link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'current_lang' not in st.session_state:
    st.session_state.current_lang = 'zh-CN'
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'result' not in st.session_state:
    st.session_state.result = {}

# --- Language Toggle Function ---
def toggle_language():
    st.session_state.current_lang = 'en' if st.session_state.current_lang == 'zh-CN' else 'zh-CN'

# --- Main App Logic ---
load_css()

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f'<div class="app-logo"><i class="ri-health-line icon"></i> <span>{t("app_title")}</span></div>', unsafe_allow_html=True)
with col2:
    if st.button(t("language")):
        toggle_language()
        st.rerun()

# --- Page Navigation ---
if st.session_state.page == 'home':
    # --- HOME PAGE (INPUT FORM) ---
    st.title(t("form_title"))

    with st.form(key='prediction_form'):
        user_inputs = {}
        # Create a 2-column layout for the form
        c1, c2 = st.columns(2)
        columns = [c1, c2, c1, c2, c1, c2]

        # Dynamically create select boxes for each feature
        for i, feature_zh in enumerate(feature_map.values()):
            with columns[i]:
                options = list(feature_options[feature_zh].keys())
                # Translate options for display
                display_options = [t(opt) for opt in options]
                
                selected_display_option = st.selectbox(
                    label=t(feature_zh),
                    options=display_options,
                    index=None,
                    placeholder=t("select_placeholder"),
                    key=f"select_{feature_zh}"
                )
                
                # Map back from translated option to original option for processing
                if selected_display_option:
                    original_option = options[display_options.index(selected_display_option)]
                    user_inputs[feature_zh] = original_option

        submitted = st.form_submit_button(t("submit_btn"), use_container_width=True)

        if submitted:
            # Check if model is loaded and all inputs are provided
            if model is None:
                st.error("Model is not loaded. Cannot proceed with prediction.")
            elif len(user_inputs) != len(feature_map):
                st.warning("Please fill in all the fields before submitting.")
            else:
                # --- DATA PROCESSING AND PREDICTION ---
                input_data = {}
                for chinese_name, chinese_value in user_inputs.items():
                    english_name = reverse_feature_map[chinese_name]
                    input_data[english_name] = feature_options[chinese_name][chinese_value]

                # Ensure features are in the correct order for the model
                features_order = list(feature_map.keys())
                df = pd.DataFrame([input_data], columns=features_order)

                # Predict probability
                proba = model.predict_proba(df)[0][1] * 100

                # Determine risk level
                if proba < 50:
                    risk_level = 'low'
                elif proba < 70:
                    risk_level = 'moderate'
                else:
                    risk_level = 'high'
                
                # Store results in session state
                st.session_state.result = {
                    "probability": f"{proba:.1f}%",
                    "risk_level_key": f"risk_level_{risk_level}",
                    "inputs": user_inputs
                }

                # Switch to the result page
                st.session_state.page = 'result'
                st.rerun()

else:
    # --- RESULT PAGE ---
    st.title(t("result_title"))
    result = st.session_state.result

    # Display the result card using HTML/CSS
    st.markdown(f"""
        <div class="result-card">
            <div class="risk-value">{result.get("probability", "N/A")}</div>
            <div class="risk-label">{t("risk_probability_label")}</div>
            <p class="explanation">{t("result_explanation")}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display the user's input summary
    st.markdown(f'<h3 class="summary-title">{t("input_summary_title")}</h3>', unsafe_allow_html=True)
    for feature_zh, value_zh in result.get("inputs", {}).items():
        st.markdown(f"""
            <div class="input-item">
                <span>{t(feature_zh)}</span>
                <strong>{t(value_zh)}</strong>
            </div>
        """, unsafe_allow_html=True)

    # "Back to Home" button
    if st.button(t("back_btn"), use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()