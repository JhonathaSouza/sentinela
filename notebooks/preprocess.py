import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_dataframe(folder_path):
    data = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.xml'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for conv in root.findall('.//conversation'):
            conv_id = conv.get('id')
            
            for msg in conv.findall('.//message'):
                author = msg.find('author').text if msg.find('author') is not None else ''
                text = msg.find('text').text if msg.find('text') is not None else ''
                
                if text:
                    data.append({
                        'conversation_id': conv_id,
                        'author_id': author,
                        'text': text
                    })
                    
    df = pd.DataFrame(data)
    return df

caminho_dos_xmls = '../data/raw/treino/'
df_conversas = xml_to_dataframe(caminho_dos_xmls)

df_conversas.to_csv('../data/processed/pan12_clean.csv', index=False)
print("Dataset convertido com sucesso!")