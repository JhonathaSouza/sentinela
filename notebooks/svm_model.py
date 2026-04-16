import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

print("1. Carregando os dados limpos...")
df = pd.read_csv('../data/processed/pan12_nlp_ready.csv')

# Remove linhas que ficaram totalmente vazias após a limpeza do texto
df = df.dropna(subset=['clean_text'])

print("2. Carregando o Gabarito (Ground Truth)...")
# ATENÇÃO: Substitua o nome abaixo pelo nome exato do arquivo .txt que tem os IDs
caminho_gabarito = '../data/raw/treino/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'

try:
    with open(caminho_gabarito, 'r') as f:
        predadores = f.read().splitlines()
    print(f"-> Foram carregados {len(predadores)} IDs de predadores conhecidos.")
except FileNotFoundError:
    print(f"ERRO: Não encontrei o gabarito em {caminho_gabarito}. Corrija o nome do arquivo!")
    exit()

print("3. Ensinando ao sistema quem é quem (1 = Predador, 0 = Normal)...")
df['is_predator'] = df['author_id'].isin(predadores).astype(int)

print("4. Transformando o texto em números (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['is_predator']

print("5. Dividindo os dados para Treino (70%) e Teste (30%)...")
# Stratify garante que a proporção de predadores seja mantida na divisão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("6. Treinando o modelo SVM (Baseline)...")
# class_weight='balanced' é o segredo aqui para dados desbalanceados (poucos predadores)
svm_model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
svm_model.fit(X_train, y_train)

print("7. Avaliando o modelo...")
y_pred = svm_model.predict(X_test)

print("\n" + "="*50)
print("🎯 RESULTADOS DO MODELO SVM")
print("="*50)
# Isso vai gerar a tabela exata com Recall, Precisão e F1-Score!
print(classification_report(y_test, y_pred, target_names=['Normal', 'Predador']))