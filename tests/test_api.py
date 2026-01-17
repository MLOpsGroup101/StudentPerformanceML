import requests
import json

# Cloud Run URL
URL = "https://stuperml-manual-709612871775.europe-west1.run.app/predict"

# Define the payload
payload = {
    "rows": [
        {
            
            "student_id": 0,
            "age": 20,
            "gender": "Female",           # Raw string (not gender_Female as api handles encoding)
            "grade_level": "2nd Year",    # Raw string
            "study_hours_per_day": 5.5,
            "attendance_percentage": 92.0,
            "study_consistency_index": 0.8,
            "sleep_hours": 7.5,
            "social_media_hours": 2.0,
            "tutoring_hours": 1.0,
            "uses_ai": 1,                 # 1 = Yes
            "ai_tools_used": "ChatGPT",   # Raw string
            "ai_usage_purpose": "Exam Prep", # Raw string
            "ai_usage_time_minutes": 45,
            "ai_dependency_score": 0.4,
            "ai_generated_content_percentage": 0.2,
            "ai_prompts_per_week": 15,
            "ai_ethics_score": 0.85,
            "last_exam_score": 82.0,
            "assignment_scores_avg": 88.5,
            "concept_understanding_score": 0.75,
            "improvement_rate": 0.05,
            "class_participation_score": 0.9,
            "passed": 1,
            "performance_category": "Medium"
        }
    ]
}

# Send Request
try:
    print(f"Sending request to {URL}...")
    response = requests.post(URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Success!")
        print(f"Predicted Final Score: {result['predictions'][0]}")
    else:
        print(f"\n❌ Error {response.status_code}")
        print("Details:", response.text)

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")