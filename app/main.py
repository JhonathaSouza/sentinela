from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator  # <-- NOVA BIBLIOTECA AQUI

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = FastAPI(title="Sentinela API", description="API para detecção de grooming online via LSTM")

print("Carregando o Cérebro do Sentinela...")
model = tf.keras.models.load_model('data/processed/sentinela_lstm.keras')
with open('data/processed/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Modelo carregado com sucesso!")

class MessageRequest(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@app.post("/api/analyze")
async def analyze_message(request: MessageRequest):
    # --- NOVA CAMADA DE TRADUÇÃO ---
    # Traduz o texto de Português (pt) para Inglês (en) automaticamente
    texto_em_ingles = GoogleTranslator(source='pt', target='en').translate(request.text)
    
    # 1. Limpa o texto (agora usando o texto que já está em inglês)
    cleaned = clean_text(texto_em_ingles)
    
    # 2. Converte para sequência numérica
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50)
    
    # 3. Pede para a IA prever
    prediction_prob = model.predict(padded)[0][0]
    
    # 4. Define o alerta
    is_predator = bool(prediction_prob > 0.5)
    
    return {
        "original_text": request.text,          # O que o usuário digitou (PT-BR)
        "translated_text": texto_em_ingles,     # O que a IA leu (EN)
        "risk_detected": is_predator,
        "risk_probability": float(prediction_prob),
        "status": "ALERTA: Possível Predador!" if is_predator else "Interação Segura."
    }