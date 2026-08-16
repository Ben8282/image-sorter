# Image Sorter

## Info

This is a local AI-powered image sorter.

First, enter the folder that contains your unsorted images.

The program loads each image and sends it through an AI image classification model. It then sorts the image into one of the categories defined in the `labels` list in `labels.py`. The actual folder names that are created come from the `folder_labels` list in `labels.py`.

If an image is classified as an animal, it is passed to a Vision Language Model (VLM) for a second, more detailed classification. The VLM uses a custom prompt (located in `classifier.py`) to determine the most appropriate animal folder, such as `rabbit`, `beaver`, or `panda`.

Both AI models run completely locally after the initial model download. No cloud API is used.

Keep in mind that AI models are not perfect, so some images may be classified incorrectly.

The faster your hardware, the faster the program will run.

---

## Dependencies

- Python
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

## Running the Program

1. Download or clone the source code.
2. Install the required dependencies.
3. Run:

```bash
python main.py
```

4. When prompted, enter the path to the folder containing your unsorted images.

If the image folder is in the same directory as the program, you can simply enter its name.


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
