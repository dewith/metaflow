"""Model Server"""

from fastapi import FastAPI
from src.models.nbow import NbowModel
from src.utils.runs import get_latest_successful_run

# Load model
last_run = get_latest_successful_run('NLPFlow', 'deployment_candidate')
nbow_model = NbowModel.from_dict(last_run.data.model_dict)

# Create FastAPI instance
app = FastAPI()


@app.get("/")  # Respond to HTTP GET at / route
def root():
    """Hello, World!"""
    return {"message": "Hello there!"}


@app.get("/sentiment")  # Respond to HTTP GET at /sentiment route
def analyze_sentiment(review: str, threshold: float = 0.5):
    """Predict sentiment of a review"""
    prediction = nbow_model.predict([review])[0]
    sentiment = "positive" if prediction > threshold else "negative"
    return {"review": review, "prediction": sentiment}
