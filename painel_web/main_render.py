import uvicorn
import os
import sys

# Garante que o Python enxergue o pacote 'app' na raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # O Render exige que a aplicação escute na porta definida na variável 'PORT'
    # Fallback para 10000 caso a variável não esteja definida
    port = int(os.environ.get("PORT", 10000))
    
    print(f"Iniciando Uvicorn na porta {port}...")
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )