from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import os

from supabase_keybert_recommendation import VolunteerRecommender as KeyBERTRecommender
from supabase_tfidf_recommendation import VolunteerRecommender as TFIDFRecommender
from llm_recommendation import llmRecommender

load_dotenv()

# Initialize app and middleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Models
# -----------------------
class RecommendRequest(BaseModel):
    userid: str

class BlurbRequest(BaseModel):
    blurb: str

# -----------------------
# Instantiate Models
# -----------------------
keybert_model = KeyBERTRecommender(supabase)
tfidf_model = TFIDFRecommender(supabase)
llm_model = llmRecommender(supabase)

# -----------------------
# Endpoints
# -----------------------

@app.post("/recommend/")
def recommend(request: RecommendRequest):
    try:
        keybert_model.fetch_data()
        keybert_model.fit()
        user_embedding = keybert_model.build_user_profile(request.userid)
        recs = keybert_model.recommend_for_user(user_embedding, top_n=1000)
        return {"jobs": recs.to_dict(orient="records")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/llm_recommend/")
def llm_recommend(request: BlurbRequest):
    try:
        llm_model.fetch_data()
        llm_model.load_data()
        recs = llm_model.recommend(request.blurb)
        return {"jobs": recs.to_dict(orient="records")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/tfidf_recommend/")
def tfidf_recommend(request: RecommendRequest):
    try:
        tfidf_model.fetch_data()
        tfidf_model.fit()
        user_text = tfidf_model.build_user_profile(request.userid)
        recs = tfidf_model.recommend_for_user(request.userid, top_n=1000)
        return {"jobs": recs.to_dict(orient="records")}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# -----------------------
# Run for Cloud Run
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))