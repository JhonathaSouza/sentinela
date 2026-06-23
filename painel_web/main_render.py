import sys
import os
# Adiciona a pasta raiz (..) ao caminho de busca do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Ajuste 'app.main:app' para o caminho real do seu FastAPI
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)