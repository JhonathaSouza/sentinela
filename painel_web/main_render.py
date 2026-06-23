import uvicorn
import os
import sys

# Garante que a raiz do projeto esteja no caminho de busca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # O Uvicorn buscará o 'app.main' com base no caminho que adicionamos ao sys.path
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)