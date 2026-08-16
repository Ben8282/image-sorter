# Image Sorter

## Info

This is a local AI-powered image sorter.

First, enter the folder that contains your unsorted images.

The program loads each image and sends it through an AI image classification model. It then sorts the image into one of the categories defined in the `labels` list in `labels.py`. The actual folder names that are created come from the `folder_labels` list in `labels.py`.

If an image is classified as an animal, it is passed to a Vision Language Model (VLM) for a second, more detailed classification. The VLM uses a custom prompt (located in `classifier.py`) to determine the most appropriate animal folder, such as `rabbit`, `beaver`, or `panda`.

Both AI models run completely locally after the initial model download. No cloud API is used.

File moving is handled by a Rust extension using PyO3.

Keep in mind that AI models are not perfect, so some images may be classified incorrectly.

The faster your hardware, the faster the program will run.

---

## Dependencies

- Python 3.12+
- Rust
- uv
- PyO3
- Pillow
- pillow-heif
- psd-tools
- rawpy
- torch
- open-clip-torch
- transformers
- huggingface-hub
- accelerate

---

## Installation

### 1. Install Rust

Rust is required because part of the program is written in Rust.

Download and install Rust using the official Rust installation instructions:

https://www.rust-lang.org/tools/install

### 2. Install uv

uv is used to manage the Python environment and dependencies.

Follow the official installation instructions:

https://docs.astral.sh/uv/getting-started/installation/

### 3. Clone the repository

Clone the repository:

```bash
git clone https://github.com/Ben8282/image-sorter.git
```

Then enter the project directory:

```bash
cd image-sorter
```

### 4. Install the Python dependencies

Run:

```bash
uv sync
```

### 5. Build the Rust code

Run:

```bash
cargo build --release
```

### 6. Run the program

Run:

```bash
uv run python main.py
```

The first time the program runs, the required AI models will be downloaded automatically.

The Qwen2.5-VL model is several gigabytes in size, so the first run can take a while.

---

## Running the Program

Start the program with:

```bash
uv run python main.py
```

The program will ask for the folder containing your unsorted images.

Enter the path to that folder.

For example:

```text
/home/user/Pictures/unsorted
```

Or, if the folder is in the same directory as the program:

```text
unsorted
```

The program will then process the images and move them into the appropriate folders.

---

## Supported Formats

Common formats:

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- GIF (.gif)

Apple formats:

- HEIF (.heif)
- HEIC (.heic)

Adobe:

- PSD (.psd)

RAW camera formats (via rawpy/LibRaw), including many manufacturers such as:

- Sony (.arw)
- Canon (.cr2, .cr3)
- Nikon (.nef)
- Fujifilm (.raf)
- Olympus (.orf)
- Panasonic (.rw2)
- Pentax (.pef)
- Leica (.rwl)
- DNG (.dng)
- And many other RAW formats supported by LibRaw.

---

## Project Structure

```text
image-sorter/
├── main.py
├── classifier.py
├── labels.py
├── pyproject.toml
├── uv.lock
├── Cargo.toml
├── Cargo.lock
├── src/
│   └── lib.rs
├── .gitignore
├── .python-version
├── LICENSE
└── README.md
```

### Python

`main.py` contains the main image-sorting program.

`classifier.py` contains the OpenCLIP and Qwen2.5-VL classification code.

`labels.py` contains the classification labels and folder names.

### Rust

`src/lib.rs` contains the Rust code used for moving files.

PyO3 is used to expose the Rust file-moving function to Python.

---

## AI Models

The project uses:

- OpenCLIP for the initial image classification.
- Qwen2.5-VL for detailed animal classification.

The models are downloaded automatically when they are first needed.

The models are not included in this repository.

---

## Hardware

The program can run on CPU or use available GPU acceleration through PyTorch.

Performance depends on the hardware and PyTorch backend available on the system.

The first model download can require several gigabytes of storage.

---

## License

This project is licensed under the Mozilla Public License 2.0.
