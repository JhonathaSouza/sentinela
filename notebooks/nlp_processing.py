import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Baixa o vocabulário de stopwords em inglês (idioma do PAN-12)
print("Baixando pacotes do NLTK...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Função para limpar, normalizar e remover stopwords."""
    if not isinstance(text, str):
        return ""
    
    # 1. Normalização: Converte tudo para letras minúsculas
    text = text.lower()
    
    # 2. Limpeza: Remove pontuações, números e caracteres especiais (deixa só letras)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenização e Remoção de Stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    # Junta as palavras novamente em uma única frase limpa
    return " ".join(words)

# --- EXECUÇÃO ---

print("1. Carregando o dataset convertido...")
df = pd.read_csv('../data/processed/pan12_clean.csv')

print("2. Iniciando a limpeza do texto (isso pode levar um minutinho)...")
# Cria uma nova coluna só com os textos limpos
df['clean_text'] = df['text'].apply(clean_text)

print("3. Iniciando a vetorização (TF-IDF)...")
# Cria o vetorizador (limitamos às 5000 palavras mais importantes para poupar memória)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_text'])

print(f"-> Sucesso! A matriz numérica foi gerada com formato: {X_tfidf.shape}")

print("4. Salvando o progresso...")
# Salva o novo DataFrame limpo na pasta processed
df.to_csv('../data/processed/pan12_nlp_ready.csv', index=False)
print("-> Arquivo 'pan12_nlp_ready.csv' salvo com sucesso!")