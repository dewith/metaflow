"""Model Server"""

from fastapi import FastAPI
from src.models.bow import NbowModel
from src.utils.runs import get_latest_successful_run

# Load model
print("Loading model")
last_run = get_latest_successful_run("NLPFlow", "deployment_candidate")
nbow_model = NbowModel.from_dict(last_run.data.model_dict)

# Test model
print("Running sanity test")
test_review = "I love this product!"  # pylint: disable=invalid-name
test_sentiment = nbow_model.predict(
    [test_review]
)  # pylint: disable=invalid-name
print(f"> Review: {test_review}")
print(f"> Predicted sentiment: {test_sentiment}")

# Create FastAPI instance
app = FastAPI()


@app.get("/")  # Respond to HTTP GET at / route
def root():
    """Hello, World!"""
    return {"message": "Hello there!"}


@app.get("/sentiment")  # Respond to HTTP GET at /sentiment route
def analyze_sentiment(review: str, threshold: float = 0.5):
    """Predict sentiment of a review"""
    sentiment = nbow_model.predict([review])
    prediction = "positive" if sentiment > threshold else "negative"
    return {
        "review": review,
        "sentiment": float(sentiment),
        "prediction": prediction,
    }
