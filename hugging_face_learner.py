# hugging_face_learner.py
# This is the complete, multi-page Streamlit application with all features integrated.

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai
import time
from gtts import gTTS
import base64
from PIL import Image
import io
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Slow Learner Prediction Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API KEY CONFIGURATION ---
# The API key has been added directly here to fix the configuration error.
try:
    # IMPORTANT: The user's API key is placed directly here.
    GOOGLE_API_KEY = "AIzaSyBQMkugdSe-sxREr1oG0PrZETuE2iLN_Sw"
    genai.configure(api_key=GOOGLE_API_KEY)
    API_KEY_CONFIGURED = True
except Exception as e:
    st.error(f"🚨 An error occurred while configuring the API key: {e}", icon="🚨")
    API_KEY_CONFIGURED = False

# --- LOAD MODELS AND DATA ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the .joblib files are in the same directory.")
        return None, None, None

model, scaler, feature_names = load_model_assets()

# --- INITIALIZE SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'Simulator'
if 'study_sessions' not in st.session_state:
    st.session_state.study_sessions = []
if 'timer_start_time' not in st.session_state:
    st.session_state.timer_start_time = None
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = []
if 'incorrect_answers' not in st.session_state:
    st.session_state.incorrect_answers = []
if 'learning_path' not in st.session_state:
    st.session_state.learning_path = []

# --- HELPER FUNCTIONS (from original app) ---
def prepare_input(user_input, cat_input, expected_features):
    df = pd.DataFrame([user_input])
    for col, value in cat_input.items():
        df[col] = value
    df = pd.get_dummies(df, drop_first=True)
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[expected_features]
    return df

def predict_support(df, scaler, model):
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return pred, prob

def calculate_risk_score(study_hours, exam_score, attendance, participation, sleep_hours, social_media, prob):
    score = 0
    if study_hours < 2:
        score += 25
    if exam_score < 65:
        score += 25
    if attendance < 85:
        score += 10
    if participation <= 2:
        score += 10
    if sleep_hours < 6:
        score += 10
    if social_media > 4:
        score += 10
    if prob > 0.35:
        score += 10
    return min(score, 100) # Ensure score doesn't exceed 100

def get_remedial_suggestions(score, exam_score, attendance_percentage, participation_rating, study_hours, sleep_hours, part_time, extracurricular):
    suggestions = []
    if score >= 50:
        suggestions.append(f"**Student may benefit from additional support (Risk Score: {score}/100)**")
        suggestions.append("---")
        suggestions.append("### ✅ General Support Recommendations")
        suggestions.extend([
            "* Meet individually to identify challenges and learning preferences.",
            "* Break down complex topics with step-by-step guidance.",
            "* Provide more practice in weaker subjects.",
            "* Incorporate visual learning aids and activities.",
            "* Encourage active participation in a supportive setting.",
            "* Recommend mentoring or peer learning sessions."
        ])
        suggestions.append("\n---")
        suggestions.append("### 📌 Personalized Observations & Strategies")
        triggered_specific = False
        if exam_score < 45:
            suggestions.append("* *Observation:* Very Low Exam Score → Focus on core concepts and regular practice.")
            triggered_specific = True
        if attendance_percentage < 75:
            suggestions.append("* *Observation:* Low Attendance → Discuss reasons and promote regular class routines.")
            triggered_specific = True
        if participation_rating <= 2:
            suggestions.append("* *Observation:* Low Class Participation → Offer incentives and create non-judgmental space.")
            triggered_specific = True
        if study_hours < 2:
            suggestions.append("* *Observation:* Insufficient Study Time → Design time tables with short, focused sessions.")
            triggered_specific = True
        if sleep_hours < 6:
            suggestions.append("* *Observation:* Poor Sleep Habits → Promote healthy sleep routines (7-8 hrs/night).")
            triggered_specific = True
        if part_time == "Yes" and study_hours < 2:
            suggestions.append("* *Observation:* Work-Study Conflict → Balance workload and offer weekend sessions.")
            triggered_specific = True
        if extracurricular == "Yes" and exam_score < 60:
            suggestions.append("* *Observation:* Over-scheduled with Activities → Temporarily reduce extracurricular load.")
            triggered_specific = True
        if not triggered_specific:
            suggestions.append("* No individual issues flagged. Support focus, motivation, and consistency.")
    return suggestions

# --- GEMINI API HELPER FUNCTIONS ---
def call_gemini(prompt, is_json=False):
    if not API_KEY_CONFIGURED: return "API Key not configured."
    model = genai.GenerativeModel('gemini-2.0-flash')
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json" if is_json else "text/plain"
    )
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"An error occurred with the AI API: {e}"

def call_gemini_vision(prompt, image):
    if not API_KEY_CONFIGURED: return "API Key not configured."
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"An error occurred with the AI Vision API: {e}"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    if st.button("🎓 Simulator", use_container_width=True):
        st.session_state.page = 'Simulator'
    if st.button("⏱️ Study Log", use_container_width=True):
        st.session_state.page = 'Study Log'
    if st.button("✨ AI Tutor", use_container_width=True):
        st.session_state.page = 'AI Tutor'
    if st.button("📝 Quiz & Feedback", use_container_width=True):
        st.session_state.page = 'Quiz'
    if st.button("🗺️ Learning Path", use_container_width=True):
        st.session_state.page = 'Learning Path'
    if st.button("🎧 Just Lesson", use_container_width=True):
        st.session_state.page = 'Just Lesson'


# --- PAGE RENDERING LOGIC ---

# ==============================================================================
# SIMULATOR PAGE
# ==============================================================================
if st.session_state.page == 'Simulator':
    st.title("🎓 Slow Learner Prediction Tool")
    st.subheader("Identify students who might need additional support")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Core Student Features")
        
        study_hours = st.number_input('Study Hours Per Day', min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        social_media_hours = st.number_input('Social Media Hours', min_value=0.0, max_value=24.0, value=3.5, step=0.5)
        attendance_percentage = st.number_input('Attendance Percentage', min_value=0.0, max_value=100.0, value=70.0)
        sleep_hours = st.number_input('Sleep Hours Per Day', min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        exam_score = st.number_input('Exam Score', min_value=0.0, max_value=100.0, value=45.0)
        mental_health_rating = st.slider('Mental Health Rating (1-5)', 1, 5, 3)
        participation_rating = st.slider('Participation Rating (1-5)', 1, 5, 2)

        with st.expander("Optional Features"):
            gender = st.selectbox('Gender', ['Female', 'Male'])
            part_time_job = st.selectbox('Part-Time Job', ['No', 'Yes'])
            extracurricular = st.selectbox('Extracurricular Participation', ['No', 'Yes'])
    
    with col2:
        if st.button('Predict Support Need', use_container_width=True, type="primary"):
            if model and scaler and feature_names:
                
                user_inputs = {
                    'Study_Hours': study_hours,
                    'Social_Media_Hours': social_media_hours,
                    'Attendance_Percentage': attendance_percentage,
                    'Sleep_Hours': sleep_hours,
                    'Exam_Score': exam_score,
                    'Mental_Health_Rating': mental_health_rating,
                    'Class_Participation_Rating': participation_rating
                }
                cat_values = {
                    'Gender': gender,
                    'Part_Time_Job': part_time_job,
                    'Extracurricular_Participation': extracurricular
                }
                
                # Prepare input for the ML model
                input_df = prepare_input(user_inputs, cat_values, feature_names)
                
                # Get prediction from the ML model
                prediction, probability = predict_support(input_df, scaler, model)
                
                # Calculate the final risk score using the original rule-based function
                risk_score = calculate_risk_score(study_hours, exam_score, attendance_percentage, participation_rating, sleep_hours, social_media_hours, probability)

                if risk_score > 50:
                    st.error(f"This student may be a slow learner (Risk Score: {risk_score}/100)")
                else:
                    st.success(f"This student appears to be on track (Risk Score: {risk_score}/100)")
                
                st.progress(risk_score)

                # Get rule-based suggestions
                suggestions = get_remedial_suggestions(risk_score, exam_score, attendance_percentage, participation_rating, study_hours, sleep_hours, part_time_job, extracurricular)
                if suggestions:
                    with st.expander("💡 Suggested Remedial Actions", expanded=True):
                        for tip in suggestions:
                            st.markdown(tip)

                # Get AI-powered suggestions as an enhancement
                with st.spinner("✨ Getting additional AI-powered advice..."):
                    prompt = f"""A student has a predicted academic risk score of {risk_score}/100. Their profile is: Study hours/day: {study_hours}, Attendance: {attendance_percentage}%, Social Media/day: {social_media_hours}h, Sleep: {sleep_hours}h, Mental wellness (1-5): {mental_health_rating}. Provide 3 additional, supportive, and practical remedial strategies to help the student improve, formatted as a markdown list."""
                    ai_suggestions = call_gemini(prompt)
                    st.markdown("### 🤖 AI-Based Suggestions (Gemini)")
                    st.markdown(ai_suggestions)
            else:
                st.error("Model is not loaded. Cannot make a prediction.")

# ==============================================================================
# STUDY LOG PAGE
# ==============================================================================
elif st.session_state.page == 'Study Log':
    st.title("⏱️ Study Log")
    
    subject = st.text_input("What subject are you studying?")
    
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.timer_running:
            if st.button("Start Timer", disabled=not subject):
                st.session_state.timer_running = True
                st.session_state.timer_start_time = time.time()
                st.session_state.subject = subject
                st.rerun()
        else:
            if st.button("Stop Timer", type="primary"):
                elapsed_seconds = int(time.time() - st.session_state.timer_start_time)
                st.session_state.study_sessions.append({
                    "subject": st.session_state.subject,
                    "duration": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
                    "date": time.strftime("%Y-%m-%d")
                })
                st.session_state.timer_running = False
                st.session_state.timer_start_time = None
                st.rerun()

    with col2:
        if st.session_state.timer_running:
            elapsed = int(time.time() - st.session_state.timer_start_time)
            st.metric("Elapsed Time", time.strftime("%H:%M:%S", time.gmtime(elapsed)))
            time.sleep(1)
            st.rerun()
        else:
            st.metric("Elapsed Time", "00:00:00")
            
    st.header("Study History")
    if not st.session_state.study_sessions:
        st.write("No study sessions logged yet.")
    else:
        for session in reversed(st.session_state.study_sessions):
            st.info(f"**{session['subject']}** - {session['duration']} on {session['date']}")

# ==============================================================================
# AI TUTOR PAGE
# ==============================================================================
elif st.session_state.page == 'AI Tutor':
    st.title("✨ AI Tutor")
    st.write("Stuck on a concept? Upload an image or ask a question.")
    
    uploaded_file = st.file_uploader("Attach an image or text file", type=['png', 'jpg', 'jpeg', 'txt'])
    prompt = st.text_area("Your question:", "Explain this file or concept...")

    if st.button("Get Explanation", type="primary"):
        if not prompt and not uploaded_file:
            st.warning("Please enter a question or upload a file.")
        else:
            with st.spinner("AI is thinking..."):
                if uploaded_file:
                    if uploaded_file.type.startswith('image/'):
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Your Uploaded Image")
                        response = call_gemini_vision(prompt, image)
                    else: # Text file
                        file_content = uploaded_file.read().decode("utf-8")
                        full_prompt = f"Based on the following text, please answer the question.\n\nText:\n---\n{file_content}\n---\n\nQuestion: {prompt}"
                        response = call_gemini(full_prompt)
                else: # Text only
                    response = call_gemini(prompt)
                
                st.markdown(response)

# ==============================================================================
# QUIZ & FEEDBACK PAGE
# ==============================================================================
elif st.session_state.page == 'Quiz':
    st.title("📝 Dynamic Quiz Generator")
    
    topic = st.text_input("Enter a topic to generate a quiz:", "The Solar System")
    if st.button("Generate Quiz"):
        with st.spinner(f"Generating a 6-question quiz on '{topic}'..."):
            prompt = f"""Generate a 6-question multiple-choice quiz for a high school student on the topic of "{topic}". Each question must have exactly 4 options. Return the result as a valid JSON array of objects. Each object should have three keys: "question" (string), "options" (an array of 4 strings), and "answer" (a string that is one of the options). Do not include any text before or after the JSON array."""
            quiz_json = call_gemini(prompt, is_json=True)
            try:
                st.session_state.current_quiz = json.loads(quiz_json)
            except json.JSONDecodeError:
                st.error("The AI failed to generate a valid quiz. Please try a different topic.")
                st.session_state.current_quiz = []
    
    if st.session_state.current_quiz:
        with st.form("quiz_form"):
            user_answers = []
            for i, q in enumerate(st.session_state.current_quiz):
                options = q['options']
                answer = st.radio(f"**{i+1}. {q['question']}**", options, index=None)
                user_answers.append({"question": q['question'], "selected": answer, "correct": q['answer']})
            
            submitted = st.form_submit_button("Submit Quiz")
            if submitted:
                score = 0
                st.session_state.incorrect_answers = []
                for ans in user_answers:
                    if ans['selected'] == ans['correct']:
                        score += 1
                    else:
                        st.session_state.incorrect_answers.append(ans)
                
                st.header("Quiz Results")
                st.subheader(f"You scored: {score}/{len(st.session_state.current_quiz)}")

                if st.session_state.incorrect_answers:
                    with st.spinner("Generating feedback..."):
                        feedback_prompt = "A student took a quiz and answered the following questions incorrectly. For each one, explain why their answer was wrong and why the correct answer is right in a simple, encouraging way.\n\n" + "\n\n".join([f"Question: \"{item['question']}\"\nStudent's Answer: \"{item['selected']}\"\nCorrect Answer: \"{item['correct']}\"" for item in st.session_state.incorrect_answers])
                        feedback = call_gemini(feedback_prompt)
                        st.markdown("---")
                        st.subheader("✨ AI Feedback on Incorrect Answers")
                        st.markdown(feedback)

# ==============================================================================
# LEARNING PATH PAGE
# ==============================================================================
elif st.session_state.page == 'Learning Path':
    st.title("🗺️ Personalized Learning Path")
    
    if not st.session_state.incorrect_answers:
        st.info("Take a quiz in the 'Quiz & Feedback' section first to generate a learning path based on your results.")
    else:
        if st.button("Generate My Learning Path", type="primary"):
            with st.spinner("Creating your personalized path..."):
                prompt = f"""Based on the following incorrectly answered quiz questions, identify the core sub-topics the student needs to review. Return the result as a valid JSON array of strings, where each string is a concise topic name. Do not include any text before or after the JSON array.\n\nIncorrect Questions:\n{chr(10).join([f'- {item["question"]}' for item in st.session_state.incorrect_answers])}"""
                path_json = call_gemini(prompt, is_json=True)
                try:
                    st.session_state.learning_path = json.loads(path_json)
                except json.JSONDecodeError:
                    st.error("The AI failed to generate a valid learning path. Please try again.")
                    st.session_state.learning_path = []


    if st.session_state.learning_path:
        st.write("Here are the key topics to review based on your quiz results. Click on a topic to get a detailed explanation from the AI Tutor.")
        for topic in st.session_state.learning_path:
            if st.button(topic, use_container_width=True):
                with st.spinner(f"Generating explanation for '{topic}'..."):
                    st.markdown(call_gemini(f"Explain the concept of '{topic}' in simple terms for a high school student."))

# ==============================================================================
# JUST LESSON PAGE
# ==============================================================================
elif st.session_state.page == 'Just Lesson':
    st.title("🎧 Just Lesson: An Audio Experience")
    
    topic = st.text_input("Enter a topic for an audio lesson:", "The Water Cycle")
    if st.button("Create Lesson", type="primary"):
        with st.spinner(f"Creating a story about '{topic}'..."):
            story_prompt = f"Explain the topic of '{topic}' as an engaging and simple story for a high school student. Make it about 150-200 words long."
            story = call_gemini(story_prompt)
            
            st.markdown(story)
            
            with st.spinner("Generating audio..."):
                try:
                    tts = gTTS(story, lang='en')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3')
                except Exception as e:
                    st.error(f"Could not generate audio. Please ensure you have an internet connection. Error: {e}")
