import uvicorn
import os

# Em produção, o Render roda apenas um processo por serviço.
# Se você precisa que o Django e o FastAPI rodem juntos, o ideal 
# é que o seu Django consuma a API FastAPI como um serviço externo.
# Vamos subir o FastAPI que é o coração da sua IA:

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Ajuste 'app.main:app' para o caminho real do seu FastAPI
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)