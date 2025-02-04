# Information Retrieval Assignment - Inverted Index

## Objectives
This assignment focuses on text processing and building an inverted index from a list of documents. The key objectives are:
- Efficient text processing using **multithreading**
- Constructing an **inverted index** from a dataset
- Analyzing the **index size** and **frequent terms**
- **Saving and loading** the index file

## Tasks Implemented

### Read and Process Text Dataset
- **Read text using multithreading:** Utilized two threads to read files from `documents.zip` efficiently.
- **Tokenization:** Break text into individual words using **NLTK**.
- **Stemming:** Reduce words to their root forms using the **Porter Stemmer**.
- **Remove stop words:** Eliminate common words using the provided stop-word list (`stopwords.zip`).

### Build the Inverted Index
- Implemented a **custom inverted index** without external libraries.
- Each word maps to a list of **document IDs** where it appears.

### Display Index Size
- Compute and display the **size of the index** in **bytes** and **megabytes (MB)** for memory usage analysis.

### Display Top n Frequent Terms
- Identify and display the most **frequently occurring terms** in the dataset.
- Configure the number of displayed terms by modifying the `number` variable in `main.py`.

### Save and Load Index File
- Save the generated inverted index to a **JSON file** (`inverted_index.json`).
- Load and reuse the saved index for further processing.

### Multithreading Implementation
- **Multithreaded reading:** Speeds up file reading by using **two threads**.
- **Multithreaded preprocessing:** Tokenization, stemming, and stop-word removal are done in parallel.

---

## Requirements
Ensure you have the required **Python libraries** installed before running the code:

```bash
pip install nltk
