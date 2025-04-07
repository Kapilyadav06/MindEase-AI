import streamlit as st
import numpy as np
import joblib
import random
import google.generativeai as genai
import matplotlib.pyplot as plt

# Configure Gemini AI (Replace with actual API key)
genai.configure(api_key="AIzaSyCT32keVDgj_v55jjVfB4PibV8UEZe0DvM")

# Load Model & Scaler
model = joblib.load("stress_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set Page Configuration
st.set_page_config(page_title="🌟AI Stress Predictor", page_icon="😌", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap');
        html, body, [class*="st"] { font-family: 'Quicksand', sans-serif; font-size: 20px; }
        .stButton>button { background-color: #254161; color: white; font-size: 22px; padding: 15px; border-radius: 12px; font-weight: bold; }
        .stTextInput>div>div>input { font-size: 20px; padding: 12px; border-radius: 10px; }
        .chat-box { background-color: #ffffff; padding: 18px; border-radius: 10px; margin: 10px 0; font-size: 20px; }
        .user-chat { background-color: #254161; color: white; text-align: left; padding: 14px; border-radius: 12px; font-weight: bold; }
        .ai-chat { background-color: #2f88f1; color: white; text-align: left; padding: 14px; border-radius: 12px; font-weight: bold; }
        .expander-header { font-weight: bold; font-size: 22px; }
    </style>
""", unsafe_allow_html=True)

# Image URLs (Replace these with actual AI-generated image URLs)
banner_img = "B.png"
ai_chatbot_img = "BOT,png"

stress_images = {
    1: "https://i.imgur.com/your-low-stress-image.png",
    2: "https://i.imgur.com/your-mild-stress-image.png",
    3: "https://i.imgur.com/your-moderate-stress-image.png",
    4: "https://i.imgur.com/your-high-stress-image.png",
    5: "https://i.imgur.com/your-severe-stress-image.png"
}

# Icons for User Input Fields
icons = {
    "age": "1.png",
    "sleep": "3.png",
    "study": "2.png",
    "stress": "4.png"
}

# Emoji Enhanced Stress Level Info
stress_info = {
    1: {"desc": "🟢 **Very Low Stress** - 🌿 Relaxed & in control. Keep it up! 🧘‍♂️", "emoji": "💆‍♂️"},
    2: {"desc": "🟡 **Mild Stress** - 😊 Manageable pressure. Stay mindful! 🌼", "emoji": "☀️"},
    3: {"desc": "🟠 **Moderate Stress** - 😟 Consider taking breaks! 🍵", "emoji": "🧘"},
    4: {"desc": "🔴 **High Stress** - 😰 Impacting daily routine. Self-care needed! ❤️", "emoji": "💔"},
    5: {"desc": "🚨 **Severe Stress** - 🚑 Consider seeking professional help. 🏥", "emoji": "⚠️"}
}

# Stress Reduction Tips with Emojis
stress_suggestions = {
    1: ["🧘‍♂️ Practice deep breathing.", "🛌 Maintain a sleep schedule.", "🚶‍♂️ Go for a short walk in nature.", "🎶 Listen to soothing music."],
    2: ["🧎‍♂️ Try light yoga.", "📴 Reduce screen time before bed.", "📝 Write in a gratitude journal.", "👨‍👩‍👧‍👦 Spend time with family and friends."],
    3: ["🗣️ Talk to someone you trust.", "🎨 Engage in creative hobbies.", "☕ Cut down on caffeine & sugar.", "🤝 Join a support community."],
    4: ["🧘‍♀️ Use meditation or mindfulness apps.", "⏸️ Take frequent short breaks.", "📵 Reduce social media consumption.", "🌿 Try aromatherapy or herbal tea."],
    5: ["👩‍⚕️ Speak to a mental health professional.", "💖 Focus on self-care daily.", "📅 Create and stick to a structured routine.", "🏋️ Engage in physical exercise regularly."]
}

# Initialize session state
if "chat_active" not in st.session_state:
    st.session_state["chat_active"] = False
    st.session_state["stress_level"] = None
    st.session_state["suggestions_given"] = []
    st.session_state["chat_history"] = []

# Preprocess Input
def preprocess_input(age, study_hours, sleep_hours, overwhelmed):
    age_mapping = {"16-18": 0, "19-21": 1, "22-25": 2, "26+": 3}
    overwhelmed_mapping = {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    input_data = np.array([[age_mapping[age], study_hours, sleep_hours, overwhelmed_mapping[overwhelmed]]])
    return scaler.transform(input_data)

# AI Chat Response
def get_gemini_response(user_input):
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    response = model.generate_content(user_input)
    return response.text

# App UI
st.markdown("<h1 style='text-align: center;'>🌟 AI Stress Predictor & Chatbot</h1>", unsafe_allow_html=True)
st.image(banner_img, use_container_width=True)
st.markdown("<h2 style='text-align: center;'> 😊🌟 Everything passes, even stress! 💙✨</h2>", unsafe_allow_html=True)
st.markdown("### 📝 **Fill in Your Daily Habits**")

# User Input Columns
col1, col2 = st.columns(2)
with col1:
    st.image(icons["age"], width=100)
    age = st.selectbox("👶 Age Group", ["16-18", "19-21", "22-25", "26+"])
    
    st.image(icons["sleep"], width=100)
    sleep_hours = st.slider("💤 Sleep Hours per Night", 0, 12, 6)

with col2:
    st.image(icons["study"], width=100)
    study_hours = st.slider("📚 Study Hours per Day", 0, 12, 3)
    
    st.image(icons["stress"], width=100)
    overwhelmed = st.selectbox("😰 How often do you feel overwhelmed?", ["Rarely", "Sometimes", "Often", "Always"])

# Predict Button
if st.button("🔍 Predict Stress Level"):
    input_data = preprocess_input(age, study_hours, sleep_hours, overwhelmed)
    stress_level = int(model.predict(input_data)[0])
    st.session_state["stress_level"] = stress_level
    st.session_state["chat_active"] = True
    st.session_state["suggestions_given"] = []
    st.session_state["chat_history"] = []

    st.image(stress_images[stress_level], width=150)
    st.success(f"{stress_info[stress_level]['emoji']} {stress_info[stress_level]['desc']}")

# Chatbot Section
if st.session_state.get("chat_active", False):
    with st.expander("💬 **Chat with Your AI Wellness Assistant 🤖**"):
        st.image("BOT.png", width=120)
        selected_level = st.radio("**Your Stress Level:**", [1, 2, 3, 4, 5], index=st.session_state["stress_level"] - 1, horizontal=True)

        if st.button("💡 **Get a Relaxation Tip**"):
            available_suggestions = [s for s in stress_suggestions[selected_level] if s not in st.session_state["suggestions_given"]]
            if available_suggestions:
                suggestion = random.choice(available_suggestions)
                st.session_state["suggestions_given"].append(suggestion)
                st.info(f"✅ {suggestion}")

        user_message = st.text_input("🗨️ **Chat with AI:**")
        if st.button("📩 **Send**"):
            if user_message:
                ai_response = get_gemini_response(user_message)  # Replace with actual AI response
                st.session_state["chat_history"].append(("You", user_message))
                st.session_state["chat_history"].append(("AI", ai_response))

        for sender, message in st.session_state["chat_history"]:
            st.markdown(f"<div class='chat-box {'user-chat' if sender == 'You' else 'ai-chat'}'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)

