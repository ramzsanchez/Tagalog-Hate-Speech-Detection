from tweetokenize.tokenizer import Tokenizer
import string
import csv
from nltk.corpus import stopwords


def remove_stopwords(words):
    # Define the list of Tagalog stopwords
    with open("tagalogstopwords.txt", "r") as f:
        tagalog_stopwords = [stopword.strip() for stopword in f]

    # Define the list of English stopwords
    english_stopwords = set(stopwords.words("english"))

    # Remove the stopwords from the list of words
    filtered_words = []
    for sublist in words:
        for word in sublist:
            # Check if the word (with no punctuation) is not a stopword
            if (
                word.lower() not in tagalog_stopwords
                and word.lower() not in english_stopwords
                and word not in string.punctuation
            ):
                filtered_words.append(word)

    # Join the remaining words back into a single string
    filtered_text = " ".join(filtered_words)

    # Return the filtered text
    return filtered_text


def tokenize_tweets(tweets):
    # Create Tokenizer instance
    tokenizer = Tokenizer()

    tokens_set = tokenizer.tokenize_set(tweets)
    output = []
    for i in tokens_set:
        output.append(i)
    return output


def process_csv_file(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "a", newline="", encoding="utf-8"
    ) as output_file:
        csvreader = csv.reader(input_file)
        csvwriter = csv.writer(output_file)

        # Write column names to output file
        column_names = next(csvreader)
        csvwriter.writerow(column_names)

        # Process each row
        for row in csvreader:
            if row and row[0]:
                tweet = row[0]
                classification = row[1] if len(row) > 1 else ""

                # Fill empty label with 2
                if not classification:
                    classification = 2

                # Tokenize and remove stopwords from tweet
                words = tokenize_tweets([tweet])
                filtered_text = remove_stopwords(words)

                # Write filtered tweet to new CSV file
                csvwriter.writerow([filtered_text, classification])


# Specify the input and output file paths
input_file_path = "hatespeech-validate-dataset.csv"
output_file_path = "preprocessed-hatespeech-validate-dataset-RoBERTa.csv"

# Call the function to process the CSV file
process_csv_file(input_file_path, output_file_path)
