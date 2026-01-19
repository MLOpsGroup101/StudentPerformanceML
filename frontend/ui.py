import streamlit as st
import requests


# Cloud Run API URL
API_URL = "https://stuperml-manual-709612871775.europe-west1.run.app/predict"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.markdown("""
This app uses a Machine Learning model deployed on **Google Cloud Run** to predict 
a student's final score based on their study habits and AI usage.
""")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
with st.form("prediction_form"):
    
    st.subheader("1. Student Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        age = st.number_input("Age", 15, 30, 20)
    with col3:
        grade_level = st.selectbox("Grade Level", ["1st Year", "2nd Year", "3rd Year", "4th Year"])

    st.subheader("2. Study Habits")
    col4, col5 = st.columns(2)
    with col4:
        study_hours = st.slider("Daily Study Hours", 0.0, 15.0, 5.0, step=0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 90)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    with col5:
        prev_score = st.number_input("Last Exam Score", 0, 100, 75)
        assignment_avg = st.number_input("Assignment Average", 0, 100, 80)
        consistency = st.slider("Study Consistency (0-1)", 0.0, 1.0, 0.8)

    st.subheader("3. AI Usage 🤖")
    uses_ai = st.checkbox("Uses AI Tools?", value=True)
    
    col6, col7 = st.columns(2)
    with col6:
        ai_tool = st.selectbox("Main AI Tool", ["ChatGPT", "Claude", "Gemini", "Copilot", "None"])
        ai_purpose = st.selectbox("Primary Purpose", ["Exam Prep", "Coding", "Doubt Solving", "Notes", "None"])
    with col7:
        ai_time = st.number_input("Mins/Day on AI", 0, 300, 45)
        ai_ethics = st.slider("AI Ethics Score (0-1)", 0.0, 1.0, 0.9)

    # Submit Button
    submitted = st.form_submit_button("Predict Score")

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------
if submitted:

    payload = {
        "rows": [
            {
                # DUMMY FIELD (Required by model but not used for prediction)
                "student_id": 0, 
                "passed": 0,
                "performance_category": "Medium",

                # USER INPUTS
                "age": age,
                "gender": gender,
                "grade_level": grade_level,
                "study_hours_per_day": study_hours,
                "attendance_percentage": attendance,
                "sleep_hours": sleep_hours,
                "last_exam_score": prev_score,
                "assignment_scores_avg": assignment_avg,
                "study_consistency_index": consistency,
                
                # AI INPUTS
                "uses_ai": 1 if uses_ai else 0,
                "ai_tools_used": ai_tool,
                "ai_usage_purpose": ai_purpose,
                "ai_usage_time_minutes": ai_time,
                "ai_ethics_score": ai_ethics,

                # DEFAULTS (Filling in the minor features we didn't ask for)
                "social_media_hours": 2.0,
                "tutoring_hours": 0.0,
                "ai_dependency_score": 0.5,
                "ai_generated_content_percentage": 0.2,
                "ai_prompts_per_week": 10,
                "concept_understanding_score": 0.7,
                "improvement_rate": 0.05,
                "class_participation_score": 0.8,
            }
        ]
    }

    # Send Request to Cloud Run
    try:
        with st.spinner("Asking the AI model..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            st.balloons()
            st.success(f"### Predicted Final Score: {prediction:.2f} / 100")
            

            if prediction > 85:
                st.write("🌟 Amazing! Keep up the good work!")
            elif prediction > 60:
                st.write("👍 Good job, but maybe study a bit more?")
            else:
                st.write("⚠️ Warning: You might want to increase study efforts.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")