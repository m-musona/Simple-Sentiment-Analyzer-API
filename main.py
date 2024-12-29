from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from pathlib import Path

from analyse import init, analyse_roberta_pt, analyse_sia

# Define the data model for the received data
class DataInput(BaseModel):
    text: str
    model: str

# Create a FastAPI instance
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Path to the HTML file
HTML_FILE = Path("templates/index.html")

@app.get("/")
async def home(request: Request):
    """Endpoint to return a simple HTML page."""
    if not HTML_FILE.exists():
        raise HTTPException(status_code=404, detail="HTML file not found.")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyse-text/")
async def analyse_text(input: DataInput):
    """Endpoint to analyse text from input."""
    try:
        if input.model == "sia":
            results = analyse_sia(input.text)
            message = "Text analysed successfully with Sentiment Intensity Analyzer."
        elif input.model == "roberta":
            results = analyse_roberta_pt(input.text)
            message = "Text analysed successfully with Twitter Roberta Base Sentiment."

        # Return a response message with the results
        return JSONResponse(
            content={"message": message, "results": results},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    init()
    uvicorn.run(app)
