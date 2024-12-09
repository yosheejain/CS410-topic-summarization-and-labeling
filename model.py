
from collections import defaultdict, Counter
import math


def calculate_unigram_language_model(transcript):
    """
    Calculate the unigram language model for a given transcript.

    Args:
        transcript (str): The transcript of the text.

    Returns:
        dict: A dictionary with words as keys and their probabilities as values.
    """
    words = transcript.split()
    total_words = len(words)
    word_frequencies = Counter(words)
    return {word: freq / total_words for word, freq in word_frequencies.items()}


def calculate_scores(full_lm, segment_lm):
    """
    Calculate the score for each word based on Full-LM and Segment-LM.

    Args:
        full_lm (dict): Unigram language model for the full transcript.
        segment_lm (dict): Unigram language model for a segment.

    Returns:
        dict: A dictionary with words as keys and their scores as values.
    """
    scores = {}
    for word in segment_lm:
        probability_full_lm = full_lm.get(word, 0)
        probability_segment_lm = segment_lm[word]
        scores[word] = -probability_full_lm + probability_segment_lm
    return scores


def extract_top_n_words(scores, n=5):
    """
    Extract the top N words based on scores.

    Args:
        scores (dict): A dictionary with words as keys and their scores as values.
        n (int): The number of top words to extract.

    Returns:
        list: A list of the top N words.
    """
    return [word for word, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n]]


def process_video_segments(full_transcript, segment_transcripts, n=5):
    """
    Process the video and its segments to extract top N representative words for each segment.

    Args:
        full_transcript (str): The full transcript of the video.
        segment_transcripts (list): A list of transcripts for each segment.
        n (int): The number of top words to extract for each segment.

    Returns:
        dict: A dictionary mapping each segment to its top N descriptive words.
    """
    full_lm = calculate_unigram_language_model(full_transcript)
    segment_word_map = {}

    for i, segment in enumerate(segment_transcripts):
        segment_lm = calculate_unigram_language_model(segment)
        scores = calculate_scores(full_lm, segment_lm)
        print(scores)
        top_words = extract_top_n_words(scores, n)
        segment_word_map[f"Segment-{i+1}"] = top_words

    return segment_word_map


# Example usage

full_transcript = "the quick brown fox jumps over the lazy dog the fox is quick"
segment_transcripts = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "the fox is quick"
]

segment_word_map = process_video_segments(full_transcript, segment_transcripts)
segment_word_map
