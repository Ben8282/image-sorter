use pyo3::prelude::*;
use std::fs;
use std::path::PathBuf;
#[pyfunction]
pub fn move_file_to_folder(file_path: PathBuf, folder_name: PathBuf) -> PyResult<()> {
    // Implementation for moving file to folder
    fs::create_dir_all(&folder_name).expect("Failed to create folder");
    let file_name = file_path.file_name().expect("Failed to get file name");
    let destination = folder_name.join(file_name);
    fs::rename(file_path, destination).expect("Failed to move file");
    Ok(())
}
#[pymodule]
fn image_sorter_main(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(move_file_to_folder, m)?)?;
    Ok(())
}