# Assignment 7: Latent Dirichlet Allocation (LDA) for Topic Modeling

This assignment demonstrates **Latent Dirichlet Allocation (LDA)**, a generative probabilistic model for discovering hidden **topics** in a text corpus. LDA assumes that each document is a mixture of a small number of topics, and that each topic is a probability distribution over words. By analyzing word co-occurrence patterns, LDA surfaces the underlying thematic structure of an unlabeled document collection.

---

## Table of Contents

1. [Overview](#overview)
2. [How the Pipeline Works](#how-the-pipeline-works)
3. [Flowchart](#flowchart)
4. [Key Components](#key-components)
   - [Step 1: Import Libraries & Prepare Corpus](#step-1-import-libraries--prepare-corpus)
   - [Step 2: Text Preprocessing](#step-2-text-preprocessing)
   - [Step 3: Build Dictionary & Bag-of-Words Corpus](#step-3-build-dictionary--bag-of-words-corpus)
   - [Step 4: Train the LDA Model](#step-4-train-the-lda-model)
   - [Step 5: Inspect Topics — Top Words per Topic](#step-5-inspect-topics--top-words-per-topic)
   - [Step 6: Visualize Topic-Word Distributions](#step-6-visualize-topic-word-distributions)
   - [Step 7: Assign Topics to Documents](#step-7-assign-topics-to-documents)
5. [Example Output](#example-output)
6. [Frequently Asked Questions](#frequently-asked-questions)

---

## Overview

**Topic modeling** is the task of automatically identifying the abstract "topics" that occur in a collection of documents. **LDA** treats each document as a bag-of-words and models each word as being drawn from one of `K` latent topics, where the topic proportions are drawn from a Dirichlet prior.

| Concept | Description |
| --------------- | ------------------------------------------------------------------- |
| **Topic** | A probability distribution over the vocabulary (a theme). |
| **Document** | A mixture of topics (each word in the document is assigned to one topic). |
| **Corpus** | A collection of documents; LDA learns the topic-word matrix from it. |
| **Alpha / Beta** | Dirichlet hyperparameters controlling document-topic and topic-word sparsity. |

This notebook uses the **`gensim`** library to build, train, and inspect an LDA model on a small sample news-style corpus, covering every step from raw text to topic interpretation.

---

## How the Pipeline Works

1. A raw text corpus (list of documents) is prepared.
2. Each document is tokenized, lowercased, and cleaned (stop words and short tokens removed; optionally lemmatized).
3. A `Dictionary` mapping word IDs to tokens is built, and each document is converted to a **bag-of-words** vector.
4. An LDA model is trained on the corpus for a fixed number of topics `K`.
5. The topics are inspected by listing the highest-probability words for each topic.
6. A table visualizes the top `N` words per topic.
7. The dominant topic for each document is inferred and printed.

---

## Flowchart

```mermaid
flowchart TD
    A[Raw Text Corpus] --> B[Tokenize + Clean\nLowercase, Stopwords, Short Tokens]
    B --> C[Build Gensim Dictionary\nWord → ID Mapping]
    C --> D[Convert to Bag-of-Words\nDocument → (word_id, count) Vectors]
    D --> E[Train LDA Model\nK Topics, Alpha, Beta, Passes]
    E --> F[Inspect Topics\nTop N Words per Topic]
    F --> G[Visualize Topic-Word\nDistributions Table]
    G --> H[Assign Dominant Topic\nto Each Document]
```

---

## Key Components

### Step 1: Import Libraries & Prepare Corpus

```python
import nltk
import re
from gensim import corpora, models
from gensim.models import LdaModel
import pandas as pd

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

```python
documents = [
    "Apple launched a new iPhone with an improved camera and longer battery life.",
    "The stock market rallied today as technology shares surged on strong earnings.",
    "The new basketball season starts in October with exciting matchups.",
    "Google released an update to Android with better privacy features.",
    "The Federal Reserve announced interest rates will remain steady.",
    ...
]
```

A small sample corpus covering technology, finance, and sports is used. Each element in `documents` is one raw text document.

---

### Step 2: Text Preprocessing

```python
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 3]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]
```

Preprocessing removes noise and reduces vocabulary size:

1. **Lowercasing:** makes matching case-insensitive.
2. **Alpha filter:** removes numbers and punctuation tokens.
3. **Stop-word removal:** removes high-frequency, low-information words (`the`, `is`, `and`, ...).
4. **Length filter:** drops very short tokens (≤ 3 characters) to focus on meaningful words.

---

### Step 3: Build Dictionary & Bag-of-Words Corpus

```python
dictionary = corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
```

- **Dictionary:** maps each unique word to a unique integer ID and records corpus statistics (document frequency).
- **Bag-of-Words (BoW):** each document is represented as a sparse vector of `(word_id, count)` pairs. This is the numerical format consumed by the LDA trainer.

---

### Step 4: Train the LDA Model

```python
lda_model = LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=False
)
```

Key hyperparameters:

| Parameter | Description |
| --------------- | ------------------------------------------------------------------- |
| `num_topics` | Number of latent topics to discover (`K`). |
| `random_state` | Seed for reproducibility. |
| `passes` | Number of passes over the corpus during training (higher = better convergence). |
| `alpha` | Document-topic prior; `'auto'` lets gensim learn an asymmetric prior. |

LDA iteratively updates two matrices: the **topic-word distribution** `φ` and the **document-topic distribution** `θ` using collapsed Gibbs sampling or variational Bayes (gensim uses the latter).

---

### Step 5: Inspect Topics — Top Words per Topic

```python
topics = lda_model.print_topics(num_words=10)
for topic_id, words in topics:
    print(f"Topic {topic_id}: {words}\n")
```

`print_topics` returns the top `N` words for each topic, weighted by their probability under that topic. Each line looks like:

```
Topic 0: 0.050*"apple" + 0.035*"phone" + 0.028*"camera" + ...
```

The human interpreter labels each topic by reading its top words.

---

### Step 6: Visualize Topic-Word Distributions

```python
topic_words = []
for topic_id in range(lda_model.num_topics):
    words = lda_model.show_topic(topic_id, topn=8)
    topic_words.append([word for word, prob in words])

max_len = max(len(w) for w in topic_words)
for topic_id, words in enumerate(topic_words):
    padded = words + [""] * (max_len - len(words))
    row = f"Topic {topic_id:<3}: " + " | ".join(f"{w:<15}" for w in padded)
    print(row)
```

A tabular view prints the top words for each topic side-by-side, making it easy to compare topics at a glance.

---

### Step 7: Assign Topics to Documents

```python
for i, bow in enumerate(bow_corpus):
    doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
    dominant = max(doc_topics, key=lambda x: x[1])
    print(f"Doc {i+1}: dominant_topic={dominant[0]}  prob={dominant[1]:.4f}")
```

For each document, `get_document_topics` returns the full topic distribution. The **dominant topic** is the one with the highest probability, providing a coarse document-level label.

---

## Example Output

```
Topic 0: 0.045*"apple" + 0.032*"phone" + 0.027*"camera" + 0.022*"battery" + 0.019*"android" + ...
Topic 1: 0.051*"market" + 0.038*"stock" + 0.030*"shares" + 0.025*"earnings" + 0.021*"federal" + ...
Topic 2: 0.049*"basketball" + 0.036*"season" + 0.029*"game" + 0.024*"team" + 0.020*"matchup" + ...

Topic 0    : apple          | phone           | camera          | battery         | ...
Topic 1    : market         | stock           | shares          | earnings        | ...
Topic 2    : basketball     | season          | game            | team            | ...

Doc 1: dominant_topic=0  prob=0.8210
Doc 2: dominant_topic=1  prob=0.7934
Doc 3: dominant_topic=2  prob=0.8562
```

Topics are interpretable: Topic 0 = technology, Topic 1 = finance, Topic 2 = sports. The dominant topic assignment correctly maps each document to its thematic cluster.

---

## Frequently Asked Questions

### 1. What is the difference between LDA and other topic models?

LDA is a **Bayesian generative model** that assumes documents are generated by first picking a topic mixture, then picking words from each topic's word distribution. Alternatives include:
- **pLSA (Probabilistic Latent Semantic Analysis):** purely descriptive, no generative story.
- **NMF (Non-negative Matrix Factorization):** linear algebraic, fast but less probabilistic.
- **BERTopic / contextual embeddings:** use Transformer embeddings for richer semantics but are heavier.

LDA remains popular for its interpretability, speed, and principled Bayesian foundation.

### 2. Why do we use a bag-of-words representation?

LDA treats documents as unordered collections of words. It does not model word order or syntax; it only cares about which words co-occur within a document. The BoW representation preserves this co-occurrence information while being lightweight and fast to train.

### 3. What do `alpha` and `beta` control?

- **Alpha (`α`):** Dirichlet prior over document-topic distributions. A low `α` encourages each document to cover few topics (sharper topic mixtures). A high `α` encourages more uniform topic coverage.
- **Beta (`β`):** Dirichlet prior over topic-word distributions. A low `β` encourages each topic to focus on a small set of words (sparser, more interpretable topics).

Using `alpha='auto'` lets gensim infer an asymmetric prior from the data, which often improves topic coherence.

### 4. How do I choose the number of topics `K`?

There is no single correct `K`. Common strategies:
- **Perplexity:** lower is better; held-out likelihood on test documents.
- **Topic coherence:** measure semantic similarity of top words within each topic (e.g., C_V coherence). Higher coherence = more interpretable topics.
- **Domain knowledge:** if you know the corpus covers `K` themes, start there.

In practice, try `K = 3–20` and inspect coherence scores.

### 5. Why are stop words and short tokens removed?

Stop words (e.g., `the`, `is`, `and`) appear in every document and carry no topical signal. Short tokens (≤ 3 characters) are often fragments or noise. Removing them reduces vocabulary size and prevents them from dominating topic-word distributions.

### 6. Can LDA handle out-of-vocabulary words?

No — LDA can only assign topics to words seen during training. Words not in the dictionary are ignored. A production system would retrain or update the dictionary with new vocabulary.

### 7. How is LDA different from word embeddings (Word2Vec)?

| Aspect | LDA | Word2Vec |
| --------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Goal | Document-level topic mixtures | Word-level vector representations |
| Output | Topic-word distributions + document-topic proportions | Dense word embeddings |
| Supervision | Unsupervised | Unsupervised |
| Granularity | Corpus / document | Individual word |

LDA is for **document-level thematic analysis**; Word2Vec is for **word-level semantic similarity**.

### 8. How do I run the code?

The code is provided as a Jupyter notebook (`code.ipynb`). Open it in Jupyter Notebook/Lab and execute the cells in order. Alternatively, extract the code into a `.py` file and run it with Python 3 after installing the required dependencies:

```bash
pip install gensim nltk pandas
python code.py
```
