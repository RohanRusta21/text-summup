from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import shutil
import os
import tempfile

app = Flask(__name__)

def chunk_text(text, chunk_size=1024):
    """Divide text into chunks of specified size."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        url = request.form['url']
        model_name = request.form['model']
        
        # Create a temporary directory to store the model
        model_dir = tempfile.mkdtemp()

        try:
            # Download and load the model into the temporary directory
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_dir)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
            
            response = requests.get(url)
            if response.status_code == 200:
                page_content = response.text
                soup = BeautifulSoup(page_content, 'html.parser')
                text = ' '.join(p.get_text() for p in soup.find_all('p'))

                chunks = list(chunk_text(text))
                summaries = []
                for chunk in chunks:
                    summarized = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summarized[0]['summary_text'])

                summary = ' '.join(summaries)
            else:
                summary = "Failed to retrieve the webpage."
        
        finally:
            # Remove the temporary directory and its contents
            shutil.rmtree(model_dir)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
