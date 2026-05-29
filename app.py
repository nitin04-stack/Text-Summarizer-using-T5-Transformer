from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch
import re

# intialaization
app = FastAPI(title="Text Summarizer App",description="Text Summarization using T5",version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("saved_summary_model")
tokenzier = T5Tokenizer.from_pretrained("saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
# templating

templates = Jinja2Templates(directory=".")

# imput schema for dialogue => string
class DailogueInput(BaseModel):
  dialogue: str
def clean_text(text):
  text = re.sub(r"\r\n"," ",text)
  text = re.sub(r"\s+"," ",text)
  text = re.sub(r"<.*?>"," ",text)
  text = text.strip().lower()
  return text

def summarize_dailogue(dialogue:str)->str:
  dialogue = clean_text(dialogue)

  dialogue = "summarize: " + dialogue

  # tokenize
  inputs = tokenzier(
      dialogue,
      padding = "max_length",
      max_length = 512,
      truncation = True,
      return_tensors = "pt"
  ).to(device)

  # generate summary in the form of tokens
  model.to(device)
  targets = model.generate(
      input_ids = inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length =150,
      num_beams = 4, # to polish out outputs
      early_stopping = True
  )

  # token_ids convert to summary
  summary = tokenzier.decode(targets[0],skip_special_tokens = True)
  return summary

# API endpoints
@app.post("/summarize")
async def create_item(dialogue_input:DailogueInput):
  summary = summarize_dailogue(dialogue_input.dialogue)
  return {"summary":summary}

@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
  return templates.TemplateResponse(request,"index.html",{"request":request})

# uvicorn app:app --reload
