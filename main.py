# Copyright (c) 2026 Ben Chorev
#
# This Source Code Form is subject to the terms of the
# Mozilla Public License, v. 2.0. If a copy of the MPL was not
# distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
import pathlib
import rawpy
from PIL import Image
from pillow_heif import register_heif_opener
from psd_tools import PSDImage
from classifier import classify, classify_folder, ANIMAL_PROMPT
from image_sorter_main import move_file_to_folder
from labels import labels, folder_labels
import time
register_heif_opener()
print("welcome to the image sorter")
print("tell me the name of the folder where your unsorted images are kept")
print("and make sure that the folder that contains your unsorted is in the same folder as this code")
print("if its not put then put full path to get to your folder")
start_time = time.perf_counter()
processed = 0
unsupported = 0
while True:
    folder = input()
    folder = pathlib.Path(folder)
    if not folder.exists():
        print("please make sure you put a real folder in")
        continue
    else:
        print("folder found")
        files = [file for file in folder.iterdir() if file.is_file()]
        total_files = len(files)
        current_file = 0
        
        for file in folder.iterdir():
            current_file += 1
            print(f"[{current_file}/{total_files}] {file.name}")
            if not file.is_file():
                continue
            image = None

            try:
                with Image.open(file) as img:
                    image = img.copy()
                    print(f"{file.name} opened succesfully")
            except:
                try:
                    with rawpy.imread(str(file)) as raw:
                        image = Image.fromarray(raw.postprocess())
                    print(f"{file.name} opened with rawpy")
                except:
                    try:
                        psd = PSDImage.open(str(file))
                        image = psd.composite()
                        print(f"{file.name} opend with psd")
                    except Exception as e:
                        print("did not work")
                        print(e)
                        print(f"{file.name} is not a supported image.")
                        unsupported += 1
            if image is not None:
                processed += 1
                print(file)
                prediction, confidence = classify(image, labels, folder_labels)
                print(f"the image is predicted to be a {prediction} with a confidence of {confidence}")
                if prediction == "animal":
                    animal_prediction = classify_folder(
                    image,
                    ANIMAL_PROMPT
                    )

                    print(f"The animal folder is {animal_prediction}")

                    move_file_to_folder(
                    file,
                    f"animal/{animal_prediction}"
                    )

                    continue
                else:
                    move_file_to_folder(file, prediction)
    elapsed = time.perf_counter() - start_time

    print("\n========== Summary ==========")
    print(f"Files found: {total_files}")
    print(f"Images processed: {processed}")
    print(f"Unsupported: {unsupported}")
    print(f"Time: {elapsed:.1f} seconds")
    break
