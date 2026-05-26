import streamlit as st
import joblib
import pandas as pd
from video_detector import predict_video

# 1. Page Setup
st.set_page_config(page_title="OmniGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ OmniGuard Threat Scanner")
st.write("Paste an email, text message, or link below to scan for fraud.")

# 2. Load the Brain
@st.cache_resource
def load_brain():
    vec = joblib.load('omniguard_vectorizer.joblib')
    mod = joblib.load('omniguard_model.joblib')
    return vec, mod

vectorizer, model = load_brain()

# 3. The User Interface
user_input = st.text_area("Message Content:", height=150)

if st.button("Scan for Threats", type="primary"):
    if user_input.strip() == "":
        st.warning("Please paste some text to scan.")
    else:
        # Translate to math and predict
        math_input = vectorizer.transform([user_input])
        prediction = model.predict(math_input)[0]
        
        # Display the UI Verdict
        st.divider()
        if prediction == 1:
            st.error("### 🚨 VERDICT: THREAT DETECTED")
        else:
            st.success("### ✅ VERDICT: SAFE (Genuine Content)")
            
        # --- THE X-RAY DASHBOARD ---
        st.write("---")
        st.write("### 🧠 AI X-Ray: Inside the Math")
        st.write("See exactly which words influenced the AI's decision.")
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        bias = model.intercept_[0]
        words_in_email = math_input.nonzero()[1]
        
        scam_words = []
        safe_words = []
        total_score = 0
        
        for index in words_in_email:
            word = feature_names[index]
            weight = coefficients[index]
            tfidf_value = math_input[0, index] # <-- THE MISSING 'x' VARIABLE!
            
            # The True Math Equation
            score = weight * tfidf_value
            total_score += score
            
            if score > 0:
                scam_words.append({"Word": word, "Raw Weight (w)": round(weight, 4), "TF-IDF (x)": round(tfidf_value, 4), "True Score (w*x)": round(score, 4)})
            elif score < 0:
                safe_words.append({"Word": word, "Raw Weight (w)": round(weight, 4), "TF-IDF (x)": round(tfidf_value, 4), "True Score (w*x)": round(score, 4)})
                
        # Create visual columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("🚨 **Scam Triggers (Pushing to Threat)**")
            if scam_words:
                # Sort by the True Score now!
                df_scam = pd.DataFrame(scam_words).sort_values(by="True Score (w*x)", ascending=False)
                st.dataframe(df_scam, use_container_width=True)
            else:
                st.write("No scam words detected.")
                
        with col2:
            st.success("✅ **Trust Triggers (Pushing to Safe)**")
            if safe_words:
                df_safe = pd.DataFrame(safe_words).sort_values(by="True Score (w*x)", ascending=True)
                st.dataframe(df_safe, use_container_width=True)
            else:
                st.write("No safe words detected.")
st.header("🎥 Video Deepfake Detection")

video_file = st.file_uploader(
    "Upload a Video",
    type=["mp4"]
)

if video_file is not None:

    with open("temp_video.mp4", "wb") as f:

        f.write(video_file.read())

    if st.button("Detect Deepfake Video"):

        result, confidence = predict_video(
            "temp_video.mp4"
        )

        st.subheader(f"Prediction: {result}")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )