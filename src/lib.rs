use pyo3::prelude::*;
use std::fs;
use std::path::Path;
#[pyfunction]
pub fn move_file_to_folder(file_path: &str, folder_name: &str) -> PyResult<()> {
    // Implementation for moving file to folder
    let file_path = Path::new(file_path);
    let folder_name = Path::new(folder_name);
    fs::create_dir_all(folder_name).expect("Failed to create folder");
    let file_name = file_path.file_name().expect("Failed to get file name");
    let destination = folder_name.join(file_name);
    fs::rename(file_path, destination).expect("Failed to move file");
    Ok(())
}
