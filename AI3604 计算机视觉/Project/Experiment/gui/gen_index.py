import os
import pickle

def list_files_in_folders(parent_folder):
    all_files = []
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            files_in_folder = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    files_in_folder.append(file_path.split('/')[-2]+'/'+os.path.basename(file_path))
            all_files.append(files_in_folder)

    return all_files

parent_folder = '../dataset/script'
file_paths_list = list_files_in_folders(parent_folder)
pickle.dump(file_paths_list, open('index.pkl', 'wb'))