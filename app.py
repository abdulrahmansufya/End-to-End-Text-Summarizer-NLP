from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from textSummarizer.pipeline.prediction import PredictionPipeline
import uvicorn

app = FastAPI()

prediction_pipeline = PredictionPipeline()


# request schema
class TextRequest(BaseModel):
    text: str


@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/docs")




@app.post("/predict", tags=["Prediction"])
async def predict_route(request: TextRequest):
    try:
        summary = prediction_pipeline.predict(request.text)

        return {
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)