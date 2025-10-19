# backend/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import time
import threading
from typing import Dict, Any, Optional
import os
import json
from dotenv import load_dotenv
import openai
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
from engine.accuracy import get_headline, check_truth,get_headline_text
from fastapi.middleware.cors import CORSMiddleware
openai.api_key = OPENAI_API_KEY
# Create FastAPI app
app = FastAPI(title="Signal Game - Game Engine")

# Allow frontend JavaScript to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now (testing only)
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],   # allow any headers
)

print("loadedkey:",OPENAI_API_KEY)
# In-memory session storage
sessions_lock = threading.Lock()
sessions: Dict[str, Dict[str, Any]] = {}

# Game configuration
TOTAL_ROUNDS = 10
DEFAULT_TIMER_S = 5.0

# Request/response models
class StartSessionReq(BaseModel):
    persona: Optional[str] = "Trader"  # can be Trader, Celebrity, Academic use dropdown box on UI

class DecisionReq(BaseModel): #convert json data into object
    headline_id: str
    
    choice: str  # e.g., "endorse", "ignore", "fact_check", "buy", "sell"
    client_timestamp: Optional[float] = None  # optional for measuring latency


import json
import time
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_rationale(headline_text: str, truth: bool, confidence: float):
    """
    Use OpenAI to generate an explanation of why the headline is true or fake.
    """
    truth_label = "TRUE" if truth else "FALSE"
    prompt = (
        f"The following news headline was evaluated as {truth_label} with confidence {confidence:.2f}.\n"
        f"{headline_text}\n"
        "Explain briefly why this classification makes sense, based on the headline_text and article preview."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert news analyst who explains whether headlines are from real or fake news."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        rationale_text = response.choices[0].message.content.strip()
        return {"explanation": rationale_text}

    except Exception as e:
        return {"explanation": f"OpenAI error: {str(e)}"}


#Endpoint: Start a new session
@app.post("/api/session")
def start_session(req: StartSessionReq):
    sid = str(uuid.uuid4())  # unique session ID
    with sessions_lock:
        sessions[sid] = {
            "persona": req.persona,
            "round": 0,
            "score": 100,
            "history": [],  # stores info for each round
            "current": None  # currently active headline, dictionary that hold all information of one headline
        }
    return {"session_id": sid, "total_rounds": TOTAL_ROUNDS,}


# Endpoint: Get next headline
@app.get("/api/session/{session_id}/next")
def get_next(session_id: str, domain: Optional[str] = None):
    with sessions_lock:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if s["round"] >= TOTAL_ROUNDS:
            return {"done": True, "message": "Game finished"}

        # Ask engine stub for a headline
        h = get_headline(domain)
        now = time.time()

        # Save current headline info with timestamp
        s["current"] = {
            "headline_id": h["headline_id"],
            "issued_at": now,
            "text":h["text"],
            "timer_s": h.get("timer_s", DEFAULT_TIMER_S),
            "domain": h["domain"],
            "source": h.get("source", ""), # .get avoids KeyError if missing
            "url": h.get("url", ""),
            "article_preview": h.get("article_preview", "")
        }

    return {
            "headline_id": h["headline_id"],
            "issued_at": now,
            "text":h["text"],
            "timer_s": h.get("timer_s", DEFAULT_TIMER_S),
            "domain": h["domain"],
            "source": h.get("source", ""), # .get avoids KeyError if missing
            "url": h.get("url", ""),
            "article_preview": h.get("article_preview", "")
    }


# Compute score 
def compute_score(choice: str, truth: bool, confidence: float, persona: str):

    if truth and choice == "endorse": #if real news and choose real"
        base = 10
    elif (not truth) and choice == "ignore": # if fake news and choose fake
        base = 10  
    else:
        base = -10

    return int(base)

# Endpoint: Submit player decision
@app.post("/api/session/{session_id}/decision")
def post_decision(session_id: str, req: DecisionReq):
    with sessions_lock:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")

        current = s.get("current")
        if not current or current.get("headline_id") != req.headline_id:
            raise HTTPException(status_code=400, detail="No active headline or mismatch")

        # Check timer
        server_now = time.time()
        elapsed = server_now - current["issued_at"]
        timed_out = elapsed > current.get("timer_s", DEFAULT_TIMER_S)
        effective_choice = req.choice if not timed_out else "ignore"
        # Check truth using engine
        truth_info = check_truth(req.headline_id)
        truth = truth_info["truth"]
        confidence = truth_info.get("confidence", 0.5)
        rationale = truth_info.get("rationale", "")
        headlineInfo = get_headline_text(req.headline_id)
        headlineText = headlineInfo.get("text","Headline not found")
        headlinePreview = headlineInfo.get("article_preview","")
        combined_text = f"Headline: {headlineText}\n{headlinePreview}"
        ai_rationale = generate_ai_rationale(
            headline_text=combined_text,  # Or engine text if available
            truth=truth, confidence=confidence)

        # Compute score
        delta = compute_score(effective_choice, truth, confidence, s["persona"])
        s["score"] += delta
        s["round"] += 1

        # Save history
        entry = {
            "round": s["round"],
            "headline_id": req.headline_id,
            "choice": req.choice,
            "effective_choice": effective_choice,
            "timed_out": timed_out,
            "truth": truth,
            "confidence": confidence,
            "rationale": rationale,
            "ai_rationale": ai_rationale,
            "score_delta": delta,
            "total_score": s["score"]
        }
        s["history"].append(entry)

        # Clear current headline
        s["current"] = None
        done = s["round"] >= TOTAL_ROUNDS

    return {
        "truth": truth,
        "confidence": confidence,
        "rationale": rationale,
        "ai_rationale": ai_rationale,
        "score_delta": delta,
        "total_score": s["score"],
        "round": s["round"],
        "total_rounds": TOTAL_ROUNDS,
        "timed_out": timed_out,
        "done": done
    }

# ===============================
# Endpoint: Get session summary
# ===============================
@app.get("/api/session/{session_id}/summary")
def get_summary(session_id: str):
    with sessions_lock:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "rounds_played": s["round"],
            "total_score": s["score"],
            "history": s["history"]
        }
    
