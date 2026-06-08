import os
import copy
import glob
import numpy as np
from PIL import Image
import torch
from io import BytesIO
import base64
import json
from torchvision import transforms
from torch.utils.data import Dataset

from estimate.prompts import get_prompt

class HBWPerson(Dataset):
    def __init__(self, img_dir, model_name, set_name='val', transform=None,
                 processor=None, prompt_keys=['height_basic'], message=None, vision_processor=None,
                 min_size=None):
        self.img_dir = img_dir # path to the images folder
        self.set_name = set_name
        self.model_name = model_name
        self.image_paths = self.get_image_paths(img_dir)
        self.processor = processor
        self.prompt_keys = prompt_keys
        self.feature_names = [pk.split("_")[0] for pk in prompt_keys]
        self.message = message
        self.vision_processor = vision_processor
        self.transform = transform
        self.min_size = min_size
        self.transform = self.build_transform_from_processor(min_size)

        # Build per-person list from the images and detections
        self.build_person_list()

    def get_image_paths(self, img_dir):
        image_paths = []
        set_dir = os.path.join(img_dir, f'{self.set_name}_small_resolution')
        for root, _, files in os.walk(set_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def build_image_name(self, img_path):
        # make the image_name composed of the subject and folder paths
        set_dir = os.path.join(self.img_dir, f'{self.set_name}_small_resolution')
        subject_path = img_path.replace(set_dir, '')

        _, subject, place, name = os.path.normpath(subject_path).split(os.path.sep)
        return f"{subject}_{place}_{name.split('.')[0]}"
        

    def build_person_list(self):
        # reorganize data to have one element per person per prompt
        self.persons = []
        for img_path in self.image_paths:
            img_name = self.build_image_name(img_path)
            # only one person per image and no detection needed
            # add the person per prompt 
            for pk, fn in zip(self.prompt_keys, self.feature_names):
                self.persons.append({
                    "img_path": img_path,
                    "img_name": img_name,
                    "person_idx": 0,
                    "prompt_key": pk,
                    "feat_name": fn,
                })
    
    def organize_predictions(self, all_preds):
        """
        Organizes predictions into a per-image dictionary structure.
        """
        organized = {}
        for person in self.persons:
            img_name = person["img_name"]
            p_idx = person["person_idx"]
            if img_name not in organized:
                organized[img_name] = {}
            for fn in self.feature_names:
                if p_idx not in organized[img_name]:
                    organized[img_name][p_idx] = {}
                organized[img_name][p_idx][fn] = all_preds.get((img_name, p_idx, fn), {})
        return organized

    def crop_person(self, image, bbox):
        x1, y1, x2, y2 = bbox  # (x1, y1): top-left, (x2, y2): bottom-right
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image.crop((x1, y1, x2, y2))

    def build_transform_from_processor(self, min_size=None):
        if self.processor is None:
            return None
        transform_list = []
        size = self.processor.image_processor.size
        if size:
            if isinstance(size, dict):
                height = size.get("height")
                width = size.get("width")
                if height and width:
                    transform_list.append(transforms.Resize((height, width)))
            elif isinstance(size, int):
                transform_list.append(transforms.Resize(size))
        if min_size and len(transform_list) == 0:
            transform_list.append(transforms.Resize(min_size))
        return transforms.Compose(transform_list)

    def image_to_base64(self, image):
        img_format = image.format or "JPEG"
        buffer = BytesIO()
        image.save(buffer, format=img_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    def update_prompt(self, prompt, template, image=None):
        message = copy.deepcopy(template) # necessary for multiple prompts
        if image is not None:
            base64_img = self.image_to_base64(image)
        if isinstance(message, dict):
            text_idx = [idx for idx, i in enumerate(message["content"]) if i["type"] == "text"][0]
            image_idx = [idx for idx, i in enumerate(message["content"]) if i["type"] == "image"][0]
            message["content"][text_idx]["text"] = message["content"][text_idx]["text"].replace("{{text}}", prompt)
            if "image" in message["content"][image_idx]:
                message["content"][image_idx]["image"] = message["content"][image_idx]["image"].replace("{{img_url}}", base64_img)
        else:
            message = message.replace("{{text}}", prompt)
            message = message.replace("{{img_url}}", base64_img)
        if self.processor.chat_template:
            message = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        return message

    def prepare_inputs(self, images, prompts_or_messages):
        if self.processor is None:
            raise ValueError("Processor is not defined.")
        return self.processor(
            text=prompts_or_messages,
            images=images,
            padding=True,
            return_tensors="pt"
        )

    def collate_fn(self, batch):
        # dummy collate function to match other datasets
        return batch

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        sample = self.persons[idx]
        img_name = sample["img_name"]
        person_img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            person_img = self.transform(person_img)

        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            prompt = self.update_prompt(prompt, self.message, image=person_img)
        inputs = self.prepare_inputs([person_img], [prompt])

        return {
            "image": person_img,
            "img_name": img_name,
            "person_idx": sample["person_idx"],
            "prompt": prompt,
            "feat_name": sample["feat_name"],
            "inputs": inputs
        }
