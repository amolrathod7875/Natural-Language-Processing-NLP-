# Natural-Language-Processing-NLP-

## NLP Basics

### What is NLP?
Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, interpret, and manipulate human language. It combines computational linguistics, machine learning, and deep learning to bridge human communication and computational systems.

### Core NLP Concepts

| Concept | Description |
|---------|-------------|
| **Tokenization** | Splitting text into individual units (tokens) such as words, phrases, or sentences |
| **Stop Words** | Common words (e.g., "the", "is", "at") that are often removed to reduce noise |
| **Stemming** | Reducing words to their root/base form using heuristic rules (e.g., "running" → "run") |
| **Lemmatization** | Reducing words to their dictionary form using morphological analysis (e.g., "better" → "good") |
| **Punctuation Removal** | Eliminating punctuation marks from text for cleaner analysis |

## Project Structure

```
NLP/
├── Day_0_Introduction_to_NLP/
│   ├── tokenizer.py           # Text tokenization with NLTK
│   ├── stopwords.py           # Stop word removal demonstration
│   ├── stemming_and_lemmatization.py  # Stem vs Lemma comparison
│   ├── punctuation.py         # Punctuation removal utilities
│   └── practice.py            # Practice exercises
├── Day_1_NLP_Pipeline/
│   ├── Text_Processing/
│   │   ├── data_acquisition.py    # TMDB API data fetching and CSV export
│   │   ├── text_processing_task.ipynb
│   │   └── tmdb_action_movies.csv
│   ├── IMBD_Case_Study/
│   │   └── Case_1_IMDB.ipynb
│   ├── Short_Hand/
│   │   ├── json_convt.py      # JSON conversion utilities
│   │   └── slang.txt
│   ├── spelling_correction.ipynb
│   ├── emojis_text_cleaning.ipynb
│   ├── html_text_cleaning.ipynb
│   ├── Task_1_Speeling_Correction.md
│   ├── Task_1_Emoji_Cleaning.md
│   └── Task_1_HTML_Cleaning.md
├── Day_2_Text_Processing/
│   ├── preprocessing.ipynb      # Text preprocessing techniques
│   ├── Feature_Extraction.ipynb # Feature extraction methods
│   └── IMDB Dataset.csv
├── Day_3_Text_Classification/
│   ├── word2vec_demo.ipynb      # Word2Vec demonstration
│   ├── Game_of_thrones/
│   │   ├── word2vec.ipynb
│   │   └── data/
│   │       ├── 001ssb.txt
│   │       ├── 002ssb.txt
│   │       ├── 003ssb.txt
│   │       ├── 004ssb.txt
│   │       └── 005ssb.txt
│   └── GoogleNews-vectors-negative300.bin.gz  # Pre-trained Word2Vec model
├── Day_4_Word2Vec/
│   ├── Word2Vec.ipynb           # Word2Vec implementation
│   ├── Tf-IDF.ipynb             # TF-IDF vectorization
│   ├── Bag_Of_Words.ipynb       # Bag of Words model
│   └── IMDB Dataset.csv
└── Day_5_Parts_of_Speech_Tagging/
    └── pos-tagging.ipynb        # POS tagging with NLTK
```

## Usage

Install dependencies:
```bash
pip install nltk pandas requests
```

Run preprocessing scripts from `Day_0_Introduction_to_NLP/` to see basic NLP operations in action.

## NLP Pipeline Flowchart

```mermaid
flowchart TD
    A[Raw Text] --> B[Tokenization]
    B --> C[Stop Words Removal]
    C --> D[Stemming/Lemmatization]
    D --> E[Punctuation Removal]
    E --> F[Clean Tokens]
    F --> G[Feature Extraction]
    G --> H[Model Training/Inference]
```

## Text Processing Pipeline

```mermaid
flowchart LR
    subgraph "Preprocessing Stage"
        A[Input Text] --> B[Lowercase]
        B --> C[Tokenize]
        C --> D[Remove Stopwords]
        D --> E[Remove Punctuation]
    end

    subgraph "Normalization Stage"
        E --> F[Stemming]
        E --> G[OR Lemmatization]
        F --> H[Normalized Tokens]
        G --> H
    end

    subgraph "Output"
        H --> I[Ready for ML/DL Models]
    end
```

## Tokenization Process

```mermaid
flowchart TD
    A["Sentence: 'Hey there my name is Amol'"] --> B[word_tokenize]
    B --> C["['Hey', 'there', 'my', 'name', 'is', 'Amol']"]
    C --> D["Remove Stopwords: 'my', 'is'"]
    D --> E["['Hey', 'there', 'name', 'Amol']"]
```

## Stemming vs Lemmatization

```mermaid
flowchart LR
    A[running] --> B[PorterStemmer]
    A --> C[WordNetLemmatizer]
    B --> D[run]
    C --> E[running]

    F[organization] --> B
    F --> C
    B --> G[organ]
    C --> H[organization]

    I[better] --> B
    I --> C
    B --> J[better]
    C --> K[good]
```

## Data Acquisition Pipeline

```mermaid
flowchart TD
    A[TMDB API] --> B[search_movie]
    B --> C[Get Results Page 1-10]
    C --> D[Loop Through Movies]
    D --> E[get_movie_details + Keywords]
    E --> F[Extract: title, overview, genres, release_date, vote_average]
    F --> G[pandas DataFrame]
    G --> H[Save to CSV]
```
