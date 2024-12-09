import tensorflow as tf
import requests
import re
import os
import tqdm
import json
from youtube_transcript_api import YouTubeTranscriptApi

# Define the feature description to decode the features
feature_description = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "labels": tf.io.VarLenFeature(tf.int64),
}

# Function to parse a single example
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Function to extract YouTube ID from the 4-character ID
def get_youtube_id(four_char_id):
    base_url = "https://data.yt8m.org/2/j/i"
    sub_path = f"/{four_char_id[:2]}/{four_char_id}.js"
    url = base_url + sub_path
    
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        
        # Extract the YouTube ID using a regular expression
        match = re.search(r'i\(".*?","(.*?)"\);', response.text)
        if match:
            return match.group(1)
        else:
            return None
    except requests.exceptions.RequestException:
        return None

# Function to get transcript for a YouTube video
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception:
        return None

# Directory containing the TFRecord files
tfrecord_dir = "/Users/arpitbansal/Desktop/cs410_project/files"  # Update with your directory path
output_dir = "./output_json_files"  # Directory to save JSON files
os.makedirs(output_dir, exist_ok=True)

# Process each TFRecord file
for tfrecord_file in os.listdir(tfrecord_dir):
    if tfrecord_file.endswith(".tfrecord"):
        tfrecord_path = os.path.join(tfrecord_dir, tfrecord_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(tfrecord_file)[0]}.json")
        
        # Create a dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(_parse_function)
        
        # Extract codes and labels
        codes = [
            (example['id'].numpy().decode('utf-8'), example['labels'].values.numpy()) 
            for example in dataset
        ]
        
        # Collect data for the JSON file
        data_to_write = []
        for four_char_id, labels in tqdm.tqdm(codes, desc=f"Processing {tfrecord_file}"):
            youtube_id = get_youtube_id(four_char_id)
            if youtube_id:
                transcript = get_transcript(youtube_id)
                if transcript:
                    data_to_write.append({
                        "video_id": youtube_id,
                        "labels": labels.tolist(),
                        "transcript": transcript,
                    })
        
        # Write data to a JSON file
        with open(output_path, 'w') as json_file:
            json.dump(data_to_write, json_file, indent=4)

        print(f"Finished processing {tfrecord_file}. Data saved to {output_path}")
