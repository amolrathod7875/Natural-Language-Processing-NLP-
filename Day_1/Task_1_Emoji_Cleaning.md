# Emoji Text Cleaning

This document explains emoji processing techniques using Python's `emoji` library for natural language processing pipelines.

---

## Code Overview

```python
import emoji

text_data = "I got the job! 😭❤️🔥 But the commute is 🚗💨..."

# Remove emojis
print(f"Emoji Removed : {emoji.replace_emoji(text_data)}")

# Demojize (replace with text descriptions)
text_demojize_data = emoji.demojize(text_data)
text_demojize_data = text_demojize_data.replace(":"," ")
print(f"\nEmoji Demojize : {text_demojize_data}")
```

---

## Techniques Explained

### 1. Emoji Removal (`replace_emoji`)

Removes all emojis from text, leaving only plain text.

**Input:**
```
"I got the job! 😭❤️🔥 But the commute is 🚗💨..."
```

**Output:**
```
"I got the job!  But the commute is ..."
```

Useful when emojis add no semantic value to your NLP task.

---

### 2. Demojization (`demojize`)

Replaces emojis with their textual descriptions wrapped in colons.

**Step-by-step:**

| Step | Output |
|------|--------|
| Original | `"I got the job! 😭❤️🔥 But the commute is 🚗💨..."` |
| After `demojize()` | `"I got the job! :loudly_crying_face: :red_heart: :fire: But the commute is :car: :dashing_away:..."` |
| After `.replace(":"," ")` | `"I got the job loudly crying face red heart fire But the commute is car dashing away ..."` |

---

## Emoji Library Functions

| Function | Description |
|----------|-------------|
| `emoji.replace_emoji(text)` | Removes emojis, returns text without them |
| `emoji.demojize(text)` | Replaces emojis with `:emoji_name:` format |
| `emoji.emojize(text)` | Converts emoji shortcodes back to emojis |

---

## Use Cases

### When to Remove Emojis
- Sentiment analysis where emojis don't contribute sentiment signals
- Keyword extraction for search indexing
- Text classification where emojis are noise

### When to Demojize
- Retain emotional context for sentiment analysis
- Include emoji meaning in token-based processing
- Create features for machine learning models

---

## Alternative Approach: Using Regex

```python
import re

# Remove emojis using regex
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
```

---

## Installation

```bash
pip install emoji
```

---

## Integration with NLP Pipeline

```python
# After HTML cleaning, before tokenization
clean_text = extract_text_with_regex(raw_html)
clean_text = emoji.replace_emoji(clean_text)  # or emojize
tokens = tokenizer(clean_text)
```