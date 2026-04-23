@echo off
title Inicializador do Sentinela

echo ==================================================
echo         LIGANDO O SISTEMA SENTINELA...
echo ==================================================
echo.

echo [1/2] Ligando o motor de Inteligencia Artificial (FastAPI)...
:: O comando 'start' abre uma nova janela. O 'cmd /k' mantem ela aberta rodando o servidor.
start "Sentinela IA (FastAPI)" cmd /k "call .venv_lstm\Scripts\activate && uvicorn app.main:app --reload"

:: Uma pequena pausa de 2 segundos para dar tempo do modelo da IA carregar na memoria
timeout /t 2 /nobreak >nul

echo [2/2] Ligando a Interface Web (Django)...
start "Sentinela Web (Django)" cmd /k "call .venv_lstm\Scripts\activate && cd painel_web && python manage.py runserver 8080"

echo.
echo ==================================================
echo  TUDO PRONTO! A IA e a Tela estao conectadas.
echo  Acesse no seu navegador: http://127.0.0.1:8080/
echo ==================================================
echo.
echo Pressione qualquer tecla para fechar esta janela de inicializacao...
pause >nul