import json
import os
import re
from pathlib import Path

def load_mapping(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_in_file(file_path, mapping):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Выполняем замену для всех ключей в mapping
        for old, new in mapping.items():
            content = content.replace(old, new)
        
        # Записываем изменения обратно в файл
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, mapping_file):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return
    
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} does not exist!")
        return
    
    # Загружаем mapping
    mapping = load_mapping(mapping_file)
    print(f"Loaded mapping with {len(mapping)} rules")
    
    # Проходим по всем файлам в директории
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                replace_in_file(file_path, mapping)
    
    print("Processing complete!")

if __name__ == "__main__":
    directory_path = "./knowledge_base"
    mapping_file_path = "terms_map.json"
    
    process_directory(directory_path, mapping_file_path)