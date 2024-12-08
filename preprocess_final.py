import os
import json
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

# Initialize tools
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Initialize the Porter Stemmer
wstem = PorterStemmer()

# Define punctuations and stopwords
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stop_words = set(stopwords.words('english'))

# Define the preprocess_text function
def preprocess_text(text):
    """
    Preprocesses input text by:
    - Removing HTML tags
    - Converting to lowercase
    - Removing punctuation
    - Removing numbers
    - Removing excess whitespace
    - Tokenizing
    - Removing stopwords
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join(char for char in text if char not in punctuations)

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize (split into words)
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [wstem.stem(word) for word in words]
    
    # Return cleaned text as a single string
    return ' '.join(words)

# Folder containing all JSON files
all_files_path = "/Users/jiya/Desktop/CS 410 Proj"

# Initialize an empty list to hold all data
video_data = []

# Read all JSON files from the directory
for file_name in os.listdir(all_files_path):
    if file_name.endswith(".json"):  # Only process JSON files
        file_path = os.path.join(all_files_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            video_data.extend(data)  # Assuming each file is a list of video records

# Preprocess transcripts and collect data
preprocessed_data = []
corpus = []

for item in video_data:
    preprocessed_transcript = preprocess_text(item['transcript'])
    corpus.append(preprocessed_transcript)
    preprocessed_data.append({
        "video_id": item["video_id"],
        "cleaned_transcript": preprocessed_transcript,
        "labels": item["labels"],
        "generated_labels": None  # Placeholder for future labels
    })

# Apply TF-IDF to remove common words across all transcripts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Remove words with low TF-IDF scores
filtered_data = []
for i, item in enumerate(preprocessed_data):
    tfidf_scores = zip(vectorizer.get_feature_names_out(), X[i].toarray()[0])
    significant_words = [word for word, score in tfidf_scores if score > 0.1]
    filtered_text = ' '.join(significant_words)
    filtered_data.append({
        "video_id": item["video_id"],
        "cleaned_transcript": filtered_text,
        "labels": item["labels"],
        "generated_labels": item["generated_labels"]
    })

# Convert to DataFrame for structured representation
df = pd.DataFrame(filtered_data)

# Save the processed DataFrame to a CSV file
output_path = "/Users/jiya/Desktop/cleaned_dataset.csv"
df.to_csv(output_path, index=False)

# Print a sample to verify
print(df.head())
