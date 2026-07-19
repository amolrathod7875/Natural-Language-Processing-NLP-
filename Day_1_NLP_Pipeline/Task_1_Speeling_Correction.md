# Spelling Correction with TextBlob

This document explains automated spelling correction using the TextBlob library for NLP text preprocessing pipelines.

---

## Code Overview

```python
from textblob import TextBlob

messay_data = "I havee a goood idea for NLP projecttt"
blob = TextBlob(messay_data)
correct_text = blob.correct()

print(f"Messay Data : {messay_data}")
print(f"Correct Data : {correct_text}")
```

---

## Output

```
Messay Data : I havee a goood idea for NLP projecttt
Correct Data : I have a good idea for NLP project
```

---

## How TextBlob Spelling Correction Works

### Algorithm
TextBlob uses a **probabilistic approach** with the following steps:

1. **Tokenization** - Splits text into individual words
2. **Edit Distance** - Calculates Levenshtein distance to known dictionary words
3. **Probability Ranking** - Ranks candidates using word frequency from training corpus
4. **Selection** - Picks the most likely correct word

### Example Corrections
| Original | Corrected | Reason |
|----------|-----------|--------|
| `havee` | `have` | Double letter → single |
| `goood` | `good` | Triple letter → single |
| `projecttt` | `project` | Double 't' removed |

---

## Installation

```bash
pip install textblob
python -m textblob.download_corpora
```

---

## Limitations

| Issue | Explanation |
|-------|-------------|
| **Context Ignorant** | Doesn't consider surrounding words for correction |
| **Proper Nouns** | May "correct" names/specific terms incorrectly |
| **Multiple Errors** | Performance degrades with many errors in sequence |
| **Training Data** | Limited to words in its corpus |

---

## Alternative Libraries

| Library | Features |
|---------|----------|
| `pyspellchecker` | Faster, no ML model required |
| `symspellpy` | Fast symmetric delete spelling correction |
| `autocorrect` | Simple wrapper, aggressive correction |

---

## Custom Word List Integration

```python
from textblob import TextBlob

# Add domain-specific words
custom_words = ["NLP", "BERT", "Transformer"]
for word in custom_words:
    TextBlob.wordsets[word] = TextBlob.wordsets.get(word, set())

text = "NLP is awsome"
blob = TextBlob(text)
print(blob.correct())  # "NLP is awesome"
```

---

## Pipeline Integration

```python
# Full preprocessing pipeline example
def preprocess_text(text):
    # Stage 1: Structural cleaning
    text = extract_text_with_regex(text)
    
    # Stage 2: Emoji handling
    text = emoji.replace_emoji(text)
    
    # Stage 3: Spelling correction
    blob = TextBlob(text)
    text = str(blob.correct())
    
    # Stage 4: Tokenization & stopword removal
    tokens = tokenizer_and_filter(text)
    
    return tokens
```