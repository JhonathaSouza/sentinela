import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_dataframe(folder_path):
    data = []
    
    # Percorre todos os arquivos XML na pasta
    for filename in os.listdir(folder_path):
        if not filename.endswith('.xml'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # A estrutura do PAN12 tem <conversation> e dentro tem <message>
        for conv in root.findall('.//conversation'):
            conv_id = conv.get('id')
            
            for msg in conv.findall('.//message'):
                author = msg.find('author').text if msg.find('author') is not None else ''
                text = msg.find('text').text if msg.find('text') is not None else ''
                
                # Só adiciona se tiver texto
                if text:
                    data.append({
                        'conversation_id': conv_id,
                        'author_id': author,
                        'text': text
                    })
                    
    # Converte a lista de dicionários para um DataFrame
    df = pd.DataFrame(data)
    return df

# Executando a função
caminho_dos_xmls = '../data/raw/treino/' # Ajuste para a sua pasta
df_conversas = xml_to_dataframe(caminho_dos_xmls)

# Salva o resultado mastigado para facilitar os próximos dias
df_conversas.to_csv('../data/processed/pan12_clean.csv', index=False)
print("Dataset convertido com sucesso!")