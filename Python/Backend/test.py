from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Use GPU

long_text = """Artificial intelligence (AI) is rapidly transforming multiple industries, from healthcare to finance.
It allows computers to process and analyze data in ways that were previously impossible. 
Machine learning, a subset of AI, enables models to improve their performance over time based on experience.
This has led to breakthroughs in areas such as speech recognition, image classification, and natural language processing."""

summary = summarizer(long_text, max_length=50, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
