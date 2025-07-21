import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from landing import show_landing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score,
                             roc_curve, auc, precision_recall_curve)

# ✅ SHOW THE LANDING PAGE FIRST
show_landing()

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(
    page_title="BreastScan AI | KNN Diagnosis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================
# Custom CSS Styling
# =================

def inject_custom_css():
    st.markdown("""
    <style>
    /* Base styles for the app */
    html, body, .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to bottom right, #f8fafc, #e0f2fe);
        color: #1e293b;
    }

    h1, h2, h3, h4 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Glassmorphic Card Styles */
    .card, .prediction-card {
        background: rgba(70, 130, 180, 0.35);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        margin-bottom: 1.5rem;
    }

    .prediction-card:hover {
        transform: scale(1.02);
    }

    /* Metrics and badges */
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }

    .badge-success {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 0.4em 0.9em;
        border-radius: 999px;
        font-weight: 600;
    }

    .badge-danger {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 0.4em 0.9em;
        border-radius: 999px;
        font-weight: 600;
    }

    /* Stylish header box */
    .glass-header {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        padding: 2rem;
        color: white;
        text-align: center;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }

    /* Plot container */
    .plot-container {
        max-width: 600px;
        margin: auto;
        border-radius: 16px;
        overflow: hidden;
        background: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    /* Animated gradient title with glow */
    .app-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        background: linear-gradient(90deg, #6366f1, #ec4899, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulseText 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        color: #64748b;
        text-align: center;
        margin-top: 0.5rem;
        animation: fadeIn 2s ease forwards;
        opacity: 0;
    }

    @keyframes pulseText {
        0%, 100% { text-shadow: 0 0 8px rgba(236, 72, 153, 0.4); }
        50% { text-shadow: 0 0 16px rgba(20, 184, 166, 0.6); }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 768px) {
        .app-title {
            font-size: 2.25rem;
        }
        .subtitle {
            font-size: 1rem;
        }
    }
    </style>
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Call this once at the top of your app
inject_custom_css()

# =================
# Load Breast Cancer Data
# =================

@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Feature units
    feature_units = {
        "mean radius": "(mm)",
        "mean texture": "(units)",
        "mean perimeter": "(mm)",
        "mean area": "(mm²)",
        "mean smoothness": "(relative)",
        "mean compactness": "(relative)",
        "mean concavity": "(relative)",
        "mean concave points": "(relative)",
        "mean symmetry": "(relative)",
        "mean fractal dimension": "(relative)"
    }

    # Feature descriptions
    feature_descriptions = {
        "mean radius": "Average distance from center to points on perimeter",
        "mean texture": "Standard deviation of gray-scale values",
        "mean perimeter": "Tumor perimeter measurement",
        "mean area": "Area measurement of tumors",
        "mean smoothness": "Local variation in radius lengths",
        "mean compactness": "Perimeter² / area - 1.0",
        "mean concavity": "Severity of concave portions of contour",
        "mean concave points": "Number of concave portions of contour",
        "mean symmetry": "Tumor symmetry measurement",
        "mean fractal dimension": "Coastline approximation - 1.0"
    }

    # Feature explanations with healthy ranges
    feature_explanations = {
        "mean radius": {
            "explanation": "How wide the tumor is from its center to the edge.",
            "healthy_range": "≤ 14.0"
        },
        "mean texture": {
            "explanation": "How rough or grainy the tumor appears.",
            "healthy_range": "≤ 20.0"
        },
        "mean perimeter": {
            "explanation": "The total distance around the tumor.",
            "healthy_range": "≤ 90.0"
        },
        "mean area": {
            "explanation": "The full surface area of the tumor.",
            "healthy_range": "≤ 600"
        },
        "mean smoothness": {
            "explanation": "How smooth or uneven the edge of the tumor is.",
            "healthy_range": "≤ 0.10"
        },
        "mean compactness": {
            "explanation": "How tightly packed the tumor appears.",
            "healthy_range": "≤ 0.15"
        },
        "mean concavity": {
            "explanation": "How deeply the tumor's edges are curved inward.",
            "healthy_range": "≤ 0.15"
        },
        "mean concave points": {
            "explanation": "How many dips/curves appear on the edge.",
            "healthy_range": "≤ 0.07"
        },
        "mean symmetry": {
            "explanation": "How symmetrical the tumor is on both sides.",
            "healthy_range": "≤ 0.20"
        },
        "mean fractal dimension": {
            "explanation": "How detailed or complex the tumor edge is.",
            "healthy_range": "≤ 0.07"
        }
    }

    return X, y, data, feature_units, feature_descriptions, feature_explanations


# Load the data
X, y, data, feature_units, feature_descriptions, feature_explanations = load_data()


# =================
# Sidebar Info
# =================
with st.sidebar:
    st.markdown("## 📊 Dataset Overview")
    st.success("You're working with the **Breast Cancer Wisconsin** dataset.")

    st.markdown("""
    - **Total Samples:** 569  
    - **Features per Sample:** 30  
    - **Classification Labels:**  
        - 🟢 Benign (Non-cancerous)  
        - 🔴 Malignant (Cancerous)
    """)

    st.markdown("---")
    st.info("Explore the dataset and model results in the main panel.")

    st.markdown("### 🧬 Tumor Feature Guide")
    with st.expander("What do these features mean?"):
        for feature, simple_desc in feature_explanations.items():
            st.markdown(f"🔹 **{feature}**: {simple_desc}")

# =================
# Model Training
# =================
k = 5  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

performance = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'cm': confusion_matrix(y_test, y_pred),
    'fpr': roc_curve(y_test, y_proba)[0],
    'tpr': roc_curve(y_test, y_proba)[1],
    'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1]),
    'precision_recall': precision_recall_curve(y_test, y_proba),
    'auprc': auc(precision_recall_curve(y_test, y_proba)[1], precision_recall_curve(y_test, y_proba)[0])
}

# =================
# Header UI
# =================
st.markdown("""
<div class="glass-header">
    <h1 class="app-title">PrismOnco</h1>
    <p class="subtitle">AI-Powered Breast Cancer Diagnosis</p>
</div>
""", unsafe_allow_html=True)

# =================
# User Input
# =================
input_data = {feature: float(X[feature].mean()) for feature in X.columns}
cols = st.columns(2)
features_split = np.array_split(X.columns[:10], 2)

for i, feature_group in enumerate(features_split):
    with cols[i]:
        for feature in feature_group:
            st.markdown(f"**{feature}** {feature_units.get(feature, '')}")
            st.caption(feature_descriptions.get(feature, ''))
            input_data[feature] = st.slider(
                label=feature,
                min_value=float(X[feature].min()),
                max_value=float(X[feature].max()),
                value=float(X[feature].mean()),
                step=0.01,
                label_visibility="collapsed",
                key=feature
            )

input_df = pd.DataFrame([input_data])[X.columns]

# =================
# Run Diagnosis Button with Session State
# =================
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("🧠 Run Diagnosis"):
    st.session_state.predict_clicked = True

# =================
# Show Results if Button Clicked
# =================
if st.session_state.predict_clicked:
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.markdown("### 🔍 Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Diagnosis")
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 4px solid {'#16a34a' if prediction == 1 else '#dc2626'};">
                <span class="{'badge-success' if prediction == 1 else 'badge-danger'}">
                    {data.target_names[prediction].upper()}
                </span>
                <p style="margin-top: 0.5rem;">Confidence: <strong>{max(proba)*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Probability Breakdown")
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="color: #16a34a;">Benign: <span class="metric-value">{proba[1]*100:.1f}%</span></p>
                <p style="color: #dc2626;">Malignant: <span class="metric-value">{proba[0]*100:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)

        # Show performance tabs
        st.markdown("## 📊 Model Performance")
        tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curve"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{performance['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{performance['precision']*100:.2f}%")
            col3.metric("Recall", f"{performance['recall']*100:.2f}%")

        with tab2:
            fig_cm = px.imshow(
                performance['cm'],
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['Malignant', 'Benign'],
                y=['Malignant', 'Benign'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(title="Confusion Matrix", width=500)
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab3:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=performance['fpr'], y=performance['tpr'],
                name=f'ROC Curve (AUC = {performance["roc_auc"]:.2f})',
                mode='lines', line=dict(color='royalblue', width=3)
            ))
            fig_roc.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.update_layout(
                title="Receiver Operating Characteristic Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
else:
    st.info("⬅️ Adjust the input sliders and click **Run Diagnosis** to see prediction and model performance.")

# =================
# Footer
# =================
st.markdown("""
<hr>
<div style="text-align: center; font-size: 0.85rem; color: #64748b;">
    BreastScan AI • KNN-Based Diagnostic Tool • For research use only
</div>
""", unsafe_allow_html=True)