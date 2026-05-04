import re
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from estimate.dataloaders.DetectionDatasets import ImageCropDatasetPerson, ImageHeadCropDatasetPerson, ImageMaskedCropDatasetPerson


def load_model(model_name, device='cuda', max_tokens=128, message=None, vision_processor=None, lora_path=None):
    if 'Qwen' in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto").to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "{{text}}"},
                {"type": "image", "image": "data:image/jpeg;base64,{{img_url}}"},
            ],
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only Qwen models are supported.")

    if lora_path is not None:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    return model, processor, max_tokens, message, vision_processor


def get_dataloader(args, processor, message=None, vision_processor=None):
    if args.dataset_name == "ImageCropDetection":
        dataset = ImageCropDatasetPerson(args.preprocessed_file, args.model_name,
                                         processor=processor,
                                         prompt_keys=args.prompt_keys,
                                         message=message)
    elif args.dataset_name == "ImageHeadCropDetection":
        dataset = ImageHeadCropDatasetPerson(args.preprocessed_file, args.model_name,
                                             processor=processor,
                                             prompt_keys=args.prompt_keys,
                                             message=message)
    elif args.dataset_name == "ImageMaskedCropDetection":
        dataset = ImageMaskedCropDatasetPerson(args.preprocessed_file, args.model_name,
                                               processor=processor,
                                               prompt_keys=args.prompt_keys,
                                               message=message)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}.")

    # num_workers must be 0: collate_fn calls the Qwen processor which is not thread-safe
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn, num_workers=0)
    return dataset, dataloader


def decode_outputs(inputs, outputs, processor, model_name, feat_names):
    output_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    decoded = processor.batch_decode(
        output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [parse_prediction(pred, fn) for pred, fn in zip(decoded, feat_names)]


KNOWN_CATEGORIES = {
    "age": ["baby", "toddler", "kid", "child", "teen", "teenager", "adult", "senior"],
    "gender": ["male", "female"],
}


def parse_prediction(output: str, feat_name="age"):
    output = output.lower()
    if "assistant" in output:
        output = output.split("assistant")[-1]
    # Handle addCriterion('key', 'value') format
    quoted = re.findall(r"'(\w+)'", output)
    if quoted:
        return quoted[-1]
    cleaned = re.sub(r'[^\w\s]', '', output).strip()
    # Look for known category names in the output
    categories = KNOWN_CATEGORIES.get(feat_name, [])
    for cat in categories:
        if re.search(rf'\b{cat}\b', cleaned):
            return cat
    # Single clean word that isn't a known category
    if len(cleaned.split()) == 1:
        return cleaned if not categories else "unknown"
    match = re.search(rf'\b{feat_name}\b\s*[:\-]?\s*(\w+)', cleaned)
    return match.group(1) if match else "unknown"


def predict_image(model, max_tokens, dataset, dataloader, processor, model_name, device='cuda'):
    all_preds = {}
    for batch in tqdm(dataloader):
        inputs = batch["inputs"].to(device, torch.float16)
        feat_names = batch["feat_names"]
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        preds = decode_outputs(inputs, outputs, processor, model_name, feat_names=feat_names)
        for img_name, p_idx, feat_name, pred in zip(
                batch["img_names"], batch["person_idxs"], feat_names, preds):
            all_preds[(img_name, p_idx, feat_name)] = pred
    return dataset.organize_predictions(all_preds)
