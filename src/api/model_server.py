from fastapi import FastAPI
from metaflow import Flow
from models.bow import NbowModel

def get_latest_successful_run(flow_nm, tag):
    """Gets the latest successful run 
        for a flow with a specific tag."""
    for r in Flow(flow_nm).runs(tag):
        if r.successful: return r
    
# load model
nbow_model = NbowModel.from_dict(
    get_latest_successful_run(
        'NLPFlow', 'deployment_candidate'
    )
    .data
    .model_dict
)

# create FastAPI instance
api = FastAPI() 

# how to respond to HTTP GET at / route
@api.get("/") 
def root():
    return {"message": "Hello there!"}

# how to respond to HTTP GET at /sentiment route
@api.get("/sentiment") 
def analyze_sentiment(review: str, threshold: float = 0.5):
    prediction = nbow_model.predict([review])[0][0]
    sentiment = "positive" if prediction > threshold else "negative"
    return {"review": review, "prediction": sentiment}
