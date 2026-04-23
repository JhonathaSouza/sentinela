import requests
from django.shortcuts import render

def interface_sentinela(request):
    contexto = {}
    
    if request.method == 'POST':
        # 1. Pega o texto que o usuário digitou na caixinha
        texto_digitado = request.POST.get('texto_chat')
        
        # 2. Chama a sua API (FastAPI) que está rodando em segundo plano
        url_api = 'http://127.0.0.1:8000/api/analyze'
        
        try:
            resposta = requests.post(url_api, json={"text": texto_digitado})
            
            if resposta.status_code == 200:
                dados = resposta.json()
                contexto['resultado'] = dados
                # Ajusta a cor do alerta com base no status
                contexto['cor_alerta'] = "danger" if dados['risk_detected'] else "success"
                # Formata a probabilidade para ficar bonita na tela (ex: 95.5%)
                contexto['porcentagem'] = round(dados['risk_probability'] * 100, 1)
        except Exception as e:
            contexto['erro'] = "A API de Inteligência Artificial está offline. Verifique o servidor FastAPI."
            
    return render(request, 'sentinela/index.html', contexto)