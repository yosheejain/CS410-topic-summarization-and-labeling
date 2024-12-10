# CS410-topic-summarization-and-labeling

In this section, we describe the steps that can be used to run this code along with dependencies or assumptions.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run the Code

Navigate to the jupyter notebook named `CS_410_Software.ipynb` and run the code. 

### Part 1: Data Collection

The first part of the code is used to collect data from the YouTube API and the Youtube 8M dataset. These are stored in the `files` folder and converted into JSONs in the `output_json_files` folder.

### Part 2: Data Preprocessing

The second part of the code is used to preprocess the data. This includes removing stop words, stemming, and lemmatizing the data. The processed data is stored in the `cleaned_dataset.csv` file.

### Part 3: Model Training (Topic Summarization)

The third part of the code is used to run the model on the data. The results are stored in the `unigram_word_map.csv` file.

### Part 4: Topic Labeling

The fourth part of the code is used to label the topics using ChatGPT API. Please note that this step would require an API key from OpenAI that was not provided in the repository. The results are stored in the `results.csv` file.

### Part 5: Evaluation

The annotated labels were integers that were mapped to their topics from the dataset using `Vocabulary.csv`. Using `results.csv` and `Vocabulary.csv`, we can evaluate the model.
