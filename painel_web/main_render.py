import uvicorn
import os

# Adiciona a pasta raiz ao sys.path para conseguir importar o pacote 'app'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # O Render injeta a porta correta na variável de ambiente 'PORT'
    port = int(os.environ.get("PORT", 10000))
    
    # Executa o Uvicorn apontando para o seu arquivo app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)