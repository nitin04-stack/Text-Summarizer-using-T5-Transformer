# Text Summarizer using T5 Transformer
A deep learning-based text summarization web application built with a fine-tuned T5 transformer model, served via FastAPI with a clean web interface.

## Links
- 🌐 **Live Demo:** [text-summarizer.onrender.com](https://text-summarizer.onrender.com)
- 🤗 **Model on HuggingFace:** [nitinn04/text-summarizer-t5](https://huggingface.co/nitinn04/text-summarizer-t5)
- 📓 **Training Notebook:** Available in same repository

##  Model Details
- **Base Model:** T5 (Text-to-Text Transfer Transformer)
- **Task:** Abstractive Text Summarization
- **Fine-tuned on:** SAMSum / Custom dialogue dataset
- **Framework:** PyTorch + HuggingFace Transformers
- **Hosted on:** HuggingFace Hub (242 MB)

## Tech Stack
| Layer | Technology |
|---|---|
| Model | T5ForConditionalGeneration (HuggingFace) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + Jinja2 Templates |
| Deployment | Render.com |
| Model Hosting | HuggingFace Hub |

## How to Run Locally
### 1. Clone the repo
```bash
git clone https://github.com/nitin04-stack/Text-Summarizer-using-transformer-NLP.git
cd Text-Summarizer-using-transformer-NLP
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
uvicorn app:app --reload
```
### 4. Open in browser
```
http://localhost:8000
```
> **Note:** Model is automatically downloaded from HuggingFace Hub on first run. No manual download needed.

## Project Structure

```
Text-Summarizer/
├── app.py                  # FastAPI backend
├── index.html              # Frontend UI
├── requirements.txt        # Dependencies
├── Procfile                # Render deployment config
└── training_notebook.ipynb # Model training (Google Colab)
```

##  How It Works

```
User Input Text
      ↓
T5 Tokenizer (max_length=512)
      ↓
Fine-tuned T5 Model
      ↓
Generated Summary Tokens
      ↓
Decoded Output Text
```

## Features
- Abstractive summarization (not just copy-paste!)
- Handles long dialogues and paragraphs
- Fast API response
- Clean and simple UI
- Auto model download from HuggingFace

## Author
**Nitin** — [GitHub](https://github.com/nitin04-stack) | [HuggingFace](https://huggingface.co/nitinn04)
