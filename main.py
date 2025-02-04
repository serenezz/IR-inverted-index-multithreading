"""
Serene Zan
CSC790 Assignment 1
Implemented multithreading in the reading from file and preprocessing stage.

Instructions:
Open the terminal in the "HW01 Zan" folder.
Type "py .\main.py" on the command line.
To change the number of most frequently appeared terms, change the "number" varaiable in the main function.
"""


import zipfile
import nltk as tk
import sys
import json
import threading
tk.download("punkt")


def read_docs_number(doc_path):
    """
    Obtains the number of documents in the zip folder.
    Parameters:
    1. doc_path : str
        Path of the documents folder.
    Returns:
    1. number : int
        The number of docucuments.
    """
    with zipfile.ZipFile(doc_path, 'r') as zip:
        number = len(zip.namelist())

    return number


def read_docs(doc_path, start, end, result_list):
    """
    Reads document files from the zip folder.
    Parameters:
    1. doc_path : str
        Path of the documents folder.
    2. start : int
        The index within the folder to start reading from.
    3. end : int
        The index within the folder to terminate.
    4. result_list : list
        An empty list that will store the strings.
    Returns:
    1. result_list : list
        A list of text strings read from the docucuments.
    """
    with zipfile.ZipFile(doc_path, 'r') as zip:
        result_list.extend(zip.read(doc).decode('utf-8') for doc in zip.namelist()[start:end])

    return result_list


def read_docs_multithreading(doc_path):
    """
    Uses two threads to read from the docs.
    Parameters:
    1. doc_path : string
        The path of the zipped folder containing the docs.
    Returns:
    1. docs : list
        A list of strings from the documents.
    """
    result_1 = []
    result_2 = []

    # Split the docs.
    total = read_docs_number(doc_path)
    mid = total // 2

    # Create two threads, each processing a portion of the dataset.
    thread1 = threading.Thread(target=read_docs, args=(doc_path, 0, mid, result_1))
    thread2 = threading.Thread(target=read_docs, args=(doc_path, mid, total+1, result_2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Combine results from two threads.
    docs = result_1 + result_2
    return docs


def read_stopwords(stop_path, stop_filename):
    """
    Reads stop words from the zip folder.
    Parameters:
    1. stop_path : str
        Path of the stop words folder.
    2. stop_filename : str
        Name of the stop words file.
    Returns:
    1. stop_words : list
        A list of stop words.
    """
    with zipfile.ZipFile(stop_path, "r") as stop_zip:
        stop_words = stop_zip.read(stop_filename).decode("utf-8").splitlines()

    return stop_words


def tokenization(doc):
    """
    Tokenizes the string by calling the nltk method.
    Parameters:
    1. doc : string
        A text string.
    Returns:
    1. tokens : list
        A list of tokenzied terms.
    """
    tokens = tk.word_tokenize(doc)
    return tokens


def stemming(tokens):
    """
    Stemming the tokens to match.
    Parameters:
    1. tokens : list
        A list of tokenized terms.
    Returns:
    1. stem_tokens : list
        A list of tokenized and stemmed terms.
    """
    stem_tokens = [tk.stem.PorterStemmer().stem(word) for word in tokens]
    return stem_tokens


def remove_stop_words(tokens, stop_words):
    """
    Removing stop words from the list of tokens.
    Parameters:
    1. tokens : list
        A list of tokenized terms.
    2. stop_words : list
        A list of stop words.
    Returns:
    1. no_sw_tokens : list
        A list of tokenized, stemmed terms without stop words.
    """
    no_sw_tokens = [word for word in tokens if not word.lower() in stop_words]
    return no_sw_tokens


def preprocess_doc(docs, stop_words, result_list):
    """
    Preprocess the terms by calling functions for tokenization, stemming, and removing stop words.
    Parameters:
    1. docs : list
        A list of text string.
    2. stop_words : list
        A list of stop words.
    3. result_list : list
        An empty list to store the lists of processed tokens.
    Returns:
    1. result_list : list
        A list of tokenized, stemmed terms without stop words.
    """
    for doc in docs:
        # Tokenization.
        tokens = tokenization(doc)
        # Stemming.
        stem_tokens = stemming(tokens)
        # Removes stop words.
        processed = remove_stop_words(stem_tokens, stop_words)
        result_list.append(processed)

    return result_list


def preprocess_multithreading(docs, stop_words):
    """
    Uses two threads to preprocess the docs.
    Parameters:
    1. docs : list
        A list of text string.
    2. stop_words : list
        A list of stop words.
    Returns:
    1. processed_tokens : list
        A list of tokenized, stemmed terms without stop words.
    """
    result_1 = []
    result_2 = []

    # Split the docs.
    mid = len(docs) // 2

    # Create two threads, each processing a portion of the dataset
    thread1 = threading.Thread(target=preprocess_doc, args=(docs[:mid], stop_words, result_1))
    thread2 = threading.Thread(target=preprocess_doc, args=(docs[mid:], stop_words, result_2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Combines results from two threads.
    processed_tokens = result_1 + result_2
    return processed_tokens


def create_inverted_index(processed_tokens):
    """
    Creates the inverted index in the dictionary data structure.
    Parameters:
    1. processed_tokens : list
        A list of tokenized, stemmed terms without stop words.
    Returns:
    1. inverted_index_dict : dictionary
        The dictionary containing the tokens and their postings.
    """
    inverted_index_dict = {}
    # Iterates over the docs.
    for doc_id, tokens in enumerate(processed_tokens):
        for token in set(tokens):
            if token not in inverted_index_dict:
                inverted_index_dict[token] = []
            inverted_index_dict[token].append(doc_id)
    
    return inverted_index_dict


def display_info():
    """
    Displays course and name.
    """
    print("\n=================== CSC790-IR Homework 01 ===================")
    print("First Name: Serene")
    print("Last Name: Zan")
    print("============================================================")


def display_tokens(inverted_index_dict, number):
    """
    Displays the top n frequently appearded tokens.
    Parameters:
    1. inverted_index_dict : dictionary
        A dictionary of tokens and postings.
    2. number : int
        The number of tokens we wish to display.
    """
    sorted_tokens = sorted(inverted_index_dict.items(), key=lambda x: len(x[1]), reverse=True)[:number]
    print(f"\nThe inverted index has been created.\nImplemented multithreading in the reading from file and preprocessing stage.")
    print(f"Now displaying top {number} terms:\n")
    for token, posting in sorted_tokens:
        print(f"{token}: {len(posting)} occurrences")


def display_size(inverted_index_dict):
    """
    Prints the size of the inverted_index_dict
    Parameters:
    1. inverted_index_dict : dictionary
        The dictionary of tokens and postings.
    """
    size_bytes = sys.getsizeof(inverted_index_dict)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nSize in Bytes: {size_bytes} bytes")
    print(f"Size in MB: {size_mb:} MB")


def save_file(inverted_index_dict, save_path):
    """
    Saves the inverted index to a new file.
    Parameters:
    1. inverted_index_dict : dictionary
        The dictionary of tokens and postings.
    """
    with open(save_path, "w") as json_file:
        json.dump(inverted_index_dict, json_file)
    print(f"\nThe inverted intex is saved in the folder, named as {save_path}.")


def main():
    if __name__ == "__main__":
        doc_path = "documents.zip"
        stop_path = "stopwords.zip"
        stop_filename = "stopwords.txt"

        # Reads from files using multithreading.
        docs = read_docs_multithreading(doc_path)
        stop_words = read_stopwords(stop_path, stop_filename)

        # Preprocess the tokens using multithreading.
        processed_tokens = preprocess_multithreading(docs, stop_words)

        # Creates the dictionary for inverted index.
        inverted_index_dict = create_inverted_index(processed_tokens)

        # Can change the value assigned to "number" to display more or less tokens.
        number = 20
        display_info()
        display_tokens(inverted_index_dict, number)
        display_size(inverted_index_dict)

        # Can change the path for saving file.
        save_path = "inverted_index.json"
        save_file(inverted_index_dict, save_path)



main()