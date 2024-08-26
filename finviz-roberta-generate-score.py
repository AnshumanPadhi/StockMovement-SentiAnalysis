import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the csv file
file_path = 'finviz-2-consolidated.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

def preprocess(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

def predict_sentiment(text):
    inputs = preprocess(text)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits
    # Convert scores to probabilities
    probabilities = torch.nn.functional.softmax(scores, dim=-1)
    # Get the class with the highest probability
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment_labels = ['negative', 'neutral', 'positive']
    predicted_sentiment = sentiment_labels[predicted_class]
    return predicted_sentiment, probabilities.tolist()

# Generate sentiment scores for each data item
texts = df['symbol'].tolist()
sentiment_results = []
for text in texts:
    sentiment, scores = predict_sentiment(text)
    sentiment_results.append((sentiment, scores))

# Add the sentiment results to the dataframe
df['Predicted Sentiment'] = [result[0] for result in sentiment_results]
df['Sentiment Scores'] = [result[1] for result in sentiment_results]

# Save the dataframe with sentiment results to a new Excel file
output_file_path = 'finviz-2-consolidated-roberate-generatedscore.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)

print("Sentiment analysis completed and results saved to", output_file_path)
