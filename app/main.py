from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords

# Garante que as stopwords estejam baixadas
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = FastAPI(title="Sentinela API", description="API para detecção de grooming online via LSTM")

print("Carregando o Cérebro do Sentinela...")
# Carrega o modelo e o dicionário de palavras (ATENÇÃO AOS CAMINHOS AQUI)
model = tf.keras.models.load_model('data/processed/sentinela_lstm.keras')
with open('data/processed/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Modelo carregado com sucesso!")

# Define o formato de entrada que a API vai receber
class MessageRequest(BaseModel):
    text: str

def clean_text(text):
    """A mesma função de limpeza usada no treinamento."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@app.post("/api/analyze")
async def analyze_message(request: MessageRequest):
    # 1. Limpa o texto
    cleaned = clean_text(request.text)
    
    # 2. Converte para sequência numérica (mesmo maxlen=50 do treinamento)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50)
    
    # 3. Pede para a IA prever
    prediction_prob = model.predict(padded)[0][0]
    
    # 4. Define o alerta (Maior que 85% = Risco)
    is_predator = bool(prediction_prob > 0.85)
    
    return {
        "original_text": request.text,
        "risk_detected": is_predator,
        "risk_probability": float(prediction_prob),
        "status": "ALERTA: Possível Predador!" if is_predator else "Interação Segura."
    }