# Triton Server - Huggingface Pipelines

### classification_bart_large_mnli
- Task: `text-classification`
- Huggingface model: https://huggingface.co/facebook/bart-large-mnli
- Pipeline: `zero-shot-classification`
- Languages supported: Not mentioned / The model is trained on English data only, but on testing, shows decent capability on zero-shot classification on various languages
- Maximum Text length: 1024 tokens
- Input params: 
    - `text`: The text to classify. Can be sent as a list of multiple lines. Response will be a list of results.
    - `labels`: The labels to classify. For sake of easy to use, is passed as a string of labels, joined by `|`.
    - `multi_label`: Boolean value to flag whether there can be multiple labels or not.
- Output: 
    - `result`: A list of JSON values in the format of `label`: `score` where 0 <= `score` <= 1
- Query test: `queries/query_classifier.py`  




### ner_wikineural_multilingual
- Task: `name-entity-recognition`
- Huggingface model: https://huggingface.co/Babelscape/wikineural-multilingual-ner
- Github: https://github.com/urchade/GLiNER
- Pipeline: `ner`
- Languages supported: German, English, Spanish, French, Italian, Dutch, Polish, Portuguese, Russian
- Maximum Text length: 512 tokens
- Input params: 
    - `text`: Input text. Can be sent as a list of multiple lines.
- Output: 
    - `result`: A list of stringified JSON values containing the NER data
- Query test: `queries/query_ner_wikineural.py`


### ner_gliner_multilingual
- Task: `name-entity-recognition`
- Huggingface model: https://huggingface.co/urchade/gliner_multi
- Pipeline: `ner`
- Languages supported: Many languages, including the low resource ones. The candidate labels should match the language of text, though.
- Maximum Text length: Not mentioned
- Input params: 
    - `text`: Input text. Can be sent as a list of multiple lines.
    - `labels`: The NER types that need to be extracted. For sake of easy to use, is passed as a string of labels, joined by `|`.
- Output: 
    - `result`: A list of stringified JSON values containing the NER data [start, end, text, label]
- Query test: `queries/query_ner_zeroshot_multilingual.py`




### sentiment_distilbert_multilingual
- Task: `sentiment-analysis`
- Huggingface model: https://huggingface.co/Babelscape/wikineural-multilingual-ner
- Pipeline: `text-classification`
- Languages supported: English, Arabic, German, Spanish, French, Japanese, Chinese, Indonesian, Hindi, Italian, Malay, Portuguese
- Maximum Text length: 512 tokens
- Input params: 
    - `text`: Input text. Can be sent as a list of multiple lines.
- Output: 
    - `result`: A list of stringified JSON values containing the sentiment and its score
- Query test: `queries/query_sentiment.py`




### summarization_bart_large_cnn
- Task: `summarization`
- Huggingface model: https://huggingface.co/facebook/bart-large-cnn
- Pipeline: `summarization`
- Languages supported: English
- Maximum Text length: Not exactly mentioned
- Input params: 
    - `text`: Input text. Can be sent as a list of multiple lines.
- Output: 
    - `summary_text`: A list of summary texts containing the sentiment and its score
- Query test: `queries/query_summarizer.py`