import torch
import open_clip
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from labels import labels, folder_labels
from pathlib import Path
from huggingface_hub import snapshot_download

AI_MODELS = {
    "tiny": {
        "name": "ViT-B/32",
        "backend": "open_clip",
        "parameters": "151M",
        "accuracy": "56.2%",
        "minimum": "CPU, 4 GB RAM",
        "recommended": "2 GB VRAM, 8 GB RAM",
        "description": "Fastest option for low-end hardware.",
    },

    "small": {
        "name": "ViT-B/16 DataComp",
        "backend": "open_clip",
        "parameters": "149M",
        "accuracy": "73.5%",
        "minimum": "CPU, 8 GB RAM",
        "recommended": "4 GB VRAM, 8 GB RAM",
        "description": "Fast with much better accuracy.",
    },

    "recommended": {
        "name": "DFN ViT-L/14",
        "backend": "open_clip",
        "parameters": "428M",
        "accuracy": "82.2%",
        "minimum": "4 GB VRAM, 16 GB RAM",
        "recommended": "8 GB VRAM, 16 GB RAM",
        "description": "Best balance of speed and accuracy.",
    },

    "high": {
        "name": "SigLIP2 SO400M/16-384",
        "backend": "open_clip",
        "parameters": "900M",
        "accuracy": "84.1%",
        "minimum": "6 GB VRAM, 16 GB RAM",
        "recommended": "12 GB VRAM, 32 GB RAM",
        "description": "High accuracy for powerful hardware.",
    },

    "maximum": {
        "name": "PE-Core-bigG/14-448",
        "backend": "open_clip",
        "parameters": "1.8B",
        "accuracy": "85.4%",
        "minimum": "12 GB VRAM, 32 GB RAM",
        "recommended": "24 GB VRAM, 64 GB RAM",
        "description": "Maximum accuracy for high-end hardware.",
    },
}
CLIP_MODELS = {
    "tiny": {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
    },

    "small": {
        "model": "ViT-B-16",
        "pretrained": "datacomp_l_s1b_b8k",
    },

    "recommended": {
        "model": "hf-hub:apple/DFN2B-CLIP-ViT-L-14",
        "pretrained": None,
    },

    "high": {
        "model": "hf-hub:timm/ViT-SO400M-16-SigLIP2-384",
        "pretrained": None,
    },

    "maximum": {
        "model": "hf-hub:timm/PE-Core-bigG-14-448",
        "pretrained": None,
    },
}

def load_clip_model(model_id):
    config = CLIP_MODELS[model_id]

    if config["pretrained"] is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config["model"],
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config["model"],
            pretrained=config["pretrained"],
        )

    tokenizer = open_clip.get_tokenizer(config["model"])

    return model, preprocess, tokenizer


model = None
preprocess = None
tokenizer = None


def set_clip_model(model_id):
    global model, preprocess, tokenizer
    model, preprocess, tokenizer = load_clip_model(model_id)

MODEL_PATH = Path(__file__).parent / "models" / "Qwen2.5-VL-3B-Instruct"

dtypes = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
]

try:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

except OSError:
    print("Qwen model not found. Downloading...")

    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        local_dir=MODEL_PATH,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

for dtype in dtypes:
    try:
        print(f"Trying {dtype}...")

        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )

        print(f"Using {dtype}")
        break
    except Exception as e:
        print(f"{dtype} failed:")
        print(e)
else:
    raise RuntimeError("Could not load Qwen model with any supported precision.")

ANIMAL_PROMPT = """
You are an AI image classifier whose ONLY job is to generate the canonical folder name for an image sorter.

The image has ALREADY been classified as an animal.

Your ONLY task is to decide what folder this animal should be placed into.

Your output will be used directly as the folder name.

This is an image organization task, NOT a biological taxonomy task.

Your goal is NOT to identify the most specific species.

Your goal is to choose the folder name that a typical human would naturally organize their photos into.

====================
OUTPUT RULES
====================

- Output EXACTLY one line.
- Output ONLY the folder name.
- No explanations.
- No punctuation.
- No quotation marks.
- No markdown.
- No emojis.
- No confidence.
- No reasoning.
- No extra words.
- Lowercase only.
- Singular nouns only.
- Use common English names.

====================
NORMALIZATION
====================

Always normalize synonyms so the same type of animal always goes into the same folder.

Examples:

house cat
domestic cat
kitty
kitten
feline

↓

cat

puppy
domestic dog
canine

↓

dog

bunny

↓

rabbit

hog
boar
piglet
swine

↓

pig

calf
bull
heifer

↓

cow

foal
mare
stallion
pony

↓

horse

cock
rooster
hen
chick

↓

chicken

killer whale

↓

orca

mountain lion
puma
catamount

↓

cougar

grey wolf
gray wolf
timber wolf

↓

wolf

====================
FOLDER LOGIC
====================

Choose the level of detail that is most useful for organizing a photo collection.

Do NOT create unnecessary folders.

Prefer broader categories over tiny subcategories.

Examples:

persian cat → cat

siamese cat → cat

ragdoll cat → cat

golden retriever → dog

labrador retriever → dog

german shepherd → dog

great dane → dog

maine coon → cat

great white shark → shark

hammerhead shark → shark

tiger shark → shark

green sea turtle → turtle

red fox → fox

bald eagle → eagle

monarch butterfly → butterfly

african elephant → elephant

asian elephant → elephant

If the exact type cannot be confidently determined, return the nearest obvious category.

Examples:

unknown dog breed → dog

unknown eagle → eagle

unknown turtle → turtle

unknown butterfly → butterfly

unknown shark → shark

====================
MULTIPLE ANIMALS
====================

If multiple animals are visible, choose the largest, closest, or most central animal.

====================
UNCERTAIN IMAGES
====================

If no reasonable folder can be chosen with confidence, output exactly:

unknown

====================
FINAL RULE
====================

Your output is a folder name, not a biological classification.

Always prefer consistency over specificity.

Thousands of similar images should end up in the same folder instead of many nearly identical folders.

Output exactly one lowercase folder name.
"""
text_fetures_cache = {}
confidence_threshold = 0.5
def classify(image, label_list, folder_labels):
    label_key = tuple(label_list)
    if label_key in text_fetures_cache:
        text_features = text_fetures_cache[label_key]
    else:
        token_text = tokenizer(label_list)
        with torch.no_grad():
            text_features = model.encode_text(token_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_fetures_cache[label_key] = text_features
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        predicted_label = folder_labels[indices.item()]
        confidence = values.item()
        if confidence < confidence_threshold:
            return "other", confidence
    return predicted_label, confidence

def classify_folder(image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    folder = response.strip().lower()

    if "\n" in folder:
        folder = folder.splitlines()[0]

    return folder