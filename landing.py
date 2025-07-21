import streamlit as st

def show_landing():
    if "landing_done" not in st.session_state:
        st.session_state.landing_done = False

    if not st.session_state.landing_done:
        st.set_page_config(page_title="Welcome | PrismOnco", layout="centered")

        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@600;800&display=swap');

            .stApp {
                background: linear-gradient(135deg, #f8bbd0 0%, #f48fb1 50%, #ce93d8 100%) !important;
                font-family: 'Raleway', sans-serif !important;
            }

            .landing-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 95vh;
                padding: 2rem;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                animation: fadeIn 1.2s ease-in-out;
            }

            @keyframes fadeIn {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            .landing-title {
                font-size: 3.5rem;
                background: linear-gradient(90deg, #8e44ad, #e91e63, #00c9ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 0.8rem;
                font-weight: 800;
            }

            .landing-subtitle {
                font-size: 1.4rem;
                color: #374151;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 500;
                max-width: 700px;
            }

            .start-button {
                background: linear-gradient(90deg, #6366f1, #ec4899);
                border: none;
                color: white;
                font-size: 1.2rem;
                padding: 0.8rem 2.2rem;
                border-radius: 40px;
                cursor: pointer;
                font-weight: 600;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
                transition: all 0.3s ease-in-out;
            }

            .start-button:hover {
                transform: scale(1.05);
                background: linear-gradient(90deg, #4338ca, #be185d);
                box-shadow: 0 10px 24px rgba(0,0,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="landing-title">Welcome to PrismOnco</div>', unsafe_allow_html=True)
        st.markdown('<div class="landing-subtitle">An AI-powered Breast Cancer Prediction Tool using K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)

        if st.button("Enter App", key="enter_button"):
            st.session_state.landing_done = True

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
