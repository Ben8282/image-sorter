import pathlib
import shutil
def move_file_to_folder(file_path, folder_name):
    # Create the folder if it doesn't exist
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    
    # Move the file to the folder
    shutil.move(file_path, pathlib.Path(folder_name) / pathlib.Path(file_path).name)