import os
import datetime
import numpy as np
import faiss
import gradio as gr
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from google_auth_oauthlib.flow import InstalledAppFlow
import os

from google_auth_oauthlib.flow import InstalledAppFlow

def authenticate_google():
    flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json",  # path to your downloaded credentials file
        scopes=["https://www.googleapis.com/auth/calendar.events"],
        redirect_uri="http://localhost/"
    )

    print("Starting OAuth flow...")
    creds = flow.run_local_server(port=5000)  # or port 8080
    print("OAuth credentials:", creds.to_json())
    return creds

# Authenticate and build service
creds = authenticate_google()
service = build("calendar", "v3", credentials=creds)

# Sample Job Descriptions
job_descriptions = {
    "Python Developer": "Looking for a Python Developer with experience in Machine Learning and Data Science.",
    "Data Analyst": "Seeking a Data Analyst with expertise in SQL, Python, and visualization tools.",
    "Web Developer": "Hiring a Web Developer skilled in JavaScript, React, and backend frameworks.",
}

# Load Sentence Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert job descriptions into embeddings
job_titles = list(job_descriptions.keys())
job_embeddings = embedding_model.encode(list(job_descriptions.values())).astype(np.float32)

# Store embeddings in FAISS
dimension = job_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(job_embeddings)

# Function to schedule an interview
def schedule_interview(candidate_name, email, job_title):
    """Schedules an interview and returns the calendar link."""
    start_time = datetime.datetime.utcnow() + datetime.timedelta(days=1, hours=2)
    end_time = start_time + datetime.timedelta(hours=1)

    event = {
        "summary": f"Interview with {candidate_name} for {job_title}",
        "location": "Google Meet",
        "description": f"Interview for {job_title} position.",
        "start": {"dateTime": start_time.isoformat(), "timeZone": "UTC"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "UTC"},
        "attendees": [{"email": email}],
        "conferenceData": {"createRequest": {"requestId": "meet"}},
    }

    event = service.events().insert(calendarId="primary", body=event, conferenceDataVersion=1).execute()
    return f"✅ Interview Scheduled: {event['htmlLink']}"

# Function to find the best job match
def find_best_match(resume_text):
    """Finds the best job match using FAISS similarity search."""
    resume_embedding = embedding_model.encode([resume_text]).astype(np.float32)
    index_result = index.search(resume_embedding, 1)[1]  # Only take the second return value
    best_match = job_titles[index_result[0][0]]
    return best_match

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📝 AI-Powered Job Interview Scheduler")
    
    with gr.Row():
        resume_input = gr.Textbox(label="Paste Resume Text Here")
        job_output = gr.Textbox(label="Best Job Match", interactive=False)
    
    with gr.Row():
        candidate_name = gr.Textbox(label="Candidate Name")
        candidate_email = gr.Textbox(label="Email")
        schedule_button = gr.Button("Schedule Interview")
        result_output = gr.Textbox(label="Interview Confirmation", interactive=False)

    resume_input.change(find_best_match, inputs=resume_input, outputs=job_output)
    schedule_button.click(
        schedule_interview,
        inputs=[candidate_name, candidate_email, job_output],
        outputs=result_output
    )

if __name__ == "__main__":
    demo.launch()
