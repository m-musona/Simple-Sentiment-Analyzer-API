# Sentiment Analysis API
This project provides a FastAPI-based web service for text sentiment analysis. It allows users to send text input and choose between two sentiment analysis models for analyzing the text: **Sentiment Intensity Analyzer (SIA)** and **Twitter Roberta Base Sentiment**. The service then returns the sentiment scores for the provided text.

## Features

- Sentiment Analysis: Analyze text for sentiment using two models:
- **SIA (Sentiment Intensity Analyzer)**
- **Roberta (Twitter Roberta Base Sentiment)**
- Web Interface: Provides a simple web interface to interact with the API.
- Results: Returns sentiment scores (Negative, Neutral, Positive) for the input text.

---

## Technologies Used

- **FastAPI**: For building the web API.
- NLP Libraries:
    NLTK for Sentiment Intensity Analyzer.
- Hugging Face Transformers for Roberta sentiment analysis.
- **Pydantic**: For data validation.
- **Jinja2**: For rendering HTML templates.
- **Scipy**: For softmax normalization of sentiment scores.

---

## Setup

To run this application, follow the steps below:

### Prerequisites

Make sure you have Python 3.7+ and the following dependencies installed:
```bash
pip install fastapi uvicorn nltk scipy transformers pydantic
```

###  Installation

1. Clone the repository or download the source code.
2. Install the required dependencies.
3. Download the necessary models for the sentiment analysis:
```bash
python -m nltk.downloader vader_lexicon
```

---

## Running the Application

Once you have everything set up, you can start the FastAPI server with the following command:
```
uvicorn main:app --reload
```

The application will be accessible at **http://localhost:8000**.

---

## Endpoints

1. GET /
- Returns the main HTML page for the text analysis service.
- Accessible via a web browser at **http://localhost:8000**.

2. POST /analyse-text/
- Description: Analyzes the provided text using the selected sentiment analysis model.
- Request Body:
```json
{
  "text": "Your text to analyze",
  "model": "sia"  // or "roberta"
}
```
- Response:
```json
{
  "message": "Text analysed successfully with Sentiment Intensity Analyzer.",
  "results": {
    "Negative": 0.12,
    "Neutral": 0.68,
    "Positive": 0.20
  }
}
```
- Models Available:
"sia": **Sentiment Intensity Analyzer (SIA)**
"roberta": **Twitter Roberta Base Sentiment (RoBERTa)**

---

## Example Usage
POST Request (Example using curl):
```bash
curl -X 'POST' \
  'http://localhost:8000/analyse-text/' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I love programming!",
  "model": "sia"
}'
```

---

## Model Details
### Sentiment Intensity Analyzer (SIA):
- A lexicon and rule-based sentiment analysis tool.
- Scores the text as Negative, Neutral, or Positive.
### Roberta (Twitter Roberta Base Sentiment):
- A transformer-based model fine-tuned on Twitter sentiment data.
- Returns sentiment scores using softmax probabilities.

## Acknowledgements
- Hugging Face for the Roberta model.
- NLTK for the Sentiment Intensity Analyzer.
- FastAPI for the web framework.

---

Feel free to contribute or raise issues on the repository!




