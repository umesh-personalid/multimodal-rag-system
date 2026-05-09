import fitz
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
import os
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


#load clip model
from dotenv import load_dotenv
load_dotenv()

#set up the environment

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY missing")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") #insure you have correct format for the model

#Embedding function for text and images

def embed_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image
    inputs = clip_processor(images=image, return_tensors="pt") # type: ignore
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
        #normalize the embedding to unit vector
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.squeeze().numpy()

def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True) # type: ignore
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
        #normalize the embedding to unit vector
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding.squeeze().numpy()
    

## process pdf and extract text and images
pdf_path = "multimodal_sample.pdf"
doc = fitz.open(pdf_path)

all_docs = []
all_embeddings = []
image_data_store = {}

#text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i,page in enumerate(doc):
    text = page.get_text()
    images = page.get_images(full=True)
    
    #split text into chunks
    text_chunks = text_splitter.split_text(text)
    
    for chunk in text_chunks:
        all_docs.append(Document(page_content=chunk, metadata={"source": f"page_{i}"}))
        all_embeddings.append(embed_text(chunk))
    
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        #store the image data for later retrieval
        image_id = f"page_{i}_img_{img_index}"
        image_data_store[image_id] = image
        
        all_docs.append(Document(page_content=image_id, metadata={"source": f"page_{i}"}))
        all_embeddings.append(embed_image(image))