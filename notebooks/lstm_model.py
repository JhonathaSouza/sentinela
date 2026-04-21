import pandas as pd
import numpy as np
import pickle  # <-- Biblioteca adicionada para salvar o Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("1. Carregando os dados limpos e o Gabarito...")
df = pd.read_csv('../data/processed/pan12_nlp_ready.csv')
df = df.dropna(subset=['clean_text'])

caminho_gabarito = '../data/raw/treino/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt' 
with open(caminho_gabarito, 'r') as f:
    predadores = f.read().splitlines()

df['is_predator'] = df['author_id'].isin(predadores).astype(int)

print("2. Preparando as sequências de texto para a LSTM...")
# Diferente do SVM, a LSTM precisa das palavras em ordem
max_features = 5000 # Vocabulário máximo
maxlen = 50 # Tamanho máximo de cada mensagem (em palavras)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['clean_text'])
X_seq = tokenizer.texts_to_sequences(df['clean_text'])

# Garante que todas as mensagens tenham o mesmo tamanho matemático
X_pad = pad_sequences(X_seq, maxlen=maxlen)
y = df['is_predator'].values

print("3. Dividindo Treino e Teste...")
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.3, random_state=42, stratify=y)

print("4. Calculando pesos para dados desbalanceados...")
# Ajuda a rede neural a prestar mais atenção na minoria (os predadores)
pesos = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(pesos))

print("5. Construindo a Arquitetura da Rede Neural LSTM...")
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid')) # Saída binária (0 ou 1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("6. Iniciando o Treinamento (ATENÇÃO: Isso pode levar alguns minutos!)...")
# Usamos apenas 3 "épocas" para não fritar o seu computador
model.fit(X_train, y_train, batch_size=256, epochs=3, validation_split=0.1, class_weight=class_weights)

print("7. Avaliando a Rede Neural...")
# Como a saída é uma probabilidade, consideramos > 0.5 como Predador (1)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n" + "="*50)
print("🧠 RESULTADOS DO MODELO LSTM")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Predador']))

# ==========================================
# PARTE NOVA: CONGELANDO E SALVANDO O MODELO
# ==========================================
print("\n8. Salvando o 'Cérebro' do Modelo e o Tokenizer...")

# Salva a arquitetura e os pesos da Rede Neural treinada
model.save('../data/processed/sentinela_lstm.keras')

# Salva o dicionário de palavras (crucial para o modelo entender novas frases depois)
with open('../data/processed/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("-> Sucesso! Modelo (.keras) e Tokenizer (.pickle) salvos na pasta 'processed'.")
print("O Sentinela está pronto para ser integrado à API!")