from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_rb = None
tokenizer_rb = None
sia = None

def init():
    global model_rb, tokenizer_rb, sia
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer_rb = AutoTokenizer.from_pretrained(MODEL)
    model_rb = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sia = SentimentIntensityAnalyzer()

# SentimentIntensityAnalyzer model results
def analyse_sia(text):
    global sia
    scores_dict = sia.polarity_scores(text)
    results = {
        "Negative" : round(scores_dict["neg"], 2),
        "Neutral" : round(scores_dict["neu"], 2),
        "Positive" : round(scores_dict["pos"], 2)
    }
    return results


# Roberter model results
def analyse_roberta_pt(text):
    global model_rb, tokenizer_rb
    encoded_text = tokenizer_rb(text, return_tensors="pt")
    output = model_rb(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    results = {
        "Negative" : round(float(scores[0]), 2),
        "Neutral" : round(float(scores[1]), 2),
        "Positive" : round(float(scores[2]), 2)
    }
    return results