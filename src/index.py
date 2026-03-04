import os

def index_directory(base_path):
    index = {}
    for root, dirs, files in os.walk(base_path):
        relative_path = os.path.relpath(root, base_path)
        current_level = index
        if relative_path != ".":
            for part in relative_path.split(os.sep):
                current_level = current_level.setdefault(part, {})
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_level[file] = f.readlines()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return index