import os
import copy
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from io import BytesIO
import base64
import json
import math
from torchvision import transforms
from torch.utils.data import Dataset

from estimate.prompts import get_prompt

class RelativeHumanPerson(Dataset):
    def __init__(self, data_dir, model_name, transform=None, set_name='test',
                 processor=None, prompt_keys=['basic'], message=None, vision_processor=None,
                 min_size=None):
        self.data_dir = data_dir
        self.model_name = model_name
        self.image_paths = glob.glob(os.path.join(data_dir, "images", "*.jpg"))
        self.set_name = set_name
        self.processor = processor
        self.prompt_keys = prompt_keys
        self.feature_names = [pk.split("_")[0] for pk in prompt_keys]
        self.message = message
        self.vision_processor = vision_processor
        self.transform = transform
        self.min_size = min_size

        self.load_gt(data_dir)
        self.transform = self.build_transform_from_processor(min_size)

        # Build per-person list from the images and annotations
        self.build_person_list()

    def build_person_list(self):
        # reorganize data to have one element per person per prompt
        self.persons = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            img_data = self.annots.get(img_name, [])
            if len(img_data) == 0:
                continue
            for p_idx, p_data in enumerate(img_data):
                bbox = p_data['wb_bbox'] if 'wb_bbox' in p_data else p_data['bbox']
                # add the person per prompt
                for pk, fn in zip(self.prompt_keys, self.feature_names):
                    self.persons.append({
                        "img_path": img_path,
                        "img_name": img_name,
                        "bbox": bbox,
                        "person_idx": p_idx,
                        "prompt_key": pk,
                        "feat_name": fn,
                    })

    def load_gt(self, RH_dir):
        print('loading gts ...')
        annot_dir = os.path.join(RH_dir, f'{self.set_name}_annots.npz')
        self.annots = np.load(annot_dir, allow_pickle=True)['annots'][()]
        print(f'Loaded {sum(len(v) for v in self.annots.values())} persons from {self.set_name}.')
        # Only keep images that have annotations
        self.image_paths = [img_path for img_path in self.image_paths if os.path.basename(img_path) in self.annots]
    
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
            for feat_name in self.feature_names:
                if p_idx not in organized[img_name]:
                    organized[img_name][p_idx] = {}
                organized[img_name][p_idx][feat_name] = all_preds.get((img_name, p_idx, feat_name), {})
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

    def update_prompt(self, prompt, message, image=None):
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
        # Dummy collate function to match other datasets
        return batch

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        sample = self.persons[idx]
        img_name = sample["img_name"]
        image = Image.open(sample["img_path"]).convert("RGB")
        bbox = sample['bbox']
        person_img = self.crop_person(image, bbox)
        if self.transform:
            person_img = self.transform(person_img)
        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=person_img)

        inputs = None
        if self.processor is not None:
            inputs = self.prepare_inputs([person_img], [prompt])

        return {
            "image": person_img,
            "bbox": bbox,
            "img_name": img_name,
            "person_idx": sample["person_idx"],
            "prompt": prompt,
            "feat_name": sample["feat_name"],
            "inputs": inputs
        }

class RelativeHumanPersonHead(RelativeHumanPerson):
    def __init__(self, data_dir, model_name, transform=None, set_name='test',
                 processor=None, prompt_keys=['basic'], message=None, vision_processor=None,
                 min_size=None):
        super().__init__(data_dir, model_name, transform, set_name, processor, prompt_keys, message, vision_processor, min_size)
        self.transform = self.build_transform_from_processor(min_size)

    def load_gt(self, RH_dir):
        """Override to preferentially load ViTPose annotations."""
        print('loading gts ...')

        # Try to load ViTPose annotations first
        vitpose_annot_dir = os.path.join(RH_dir, f'{self.set_name}_annots_vitpose.npz')
        original_annot_dir = os.path.join(RH_dir, f'{self.set_name}_annots.npz')

        if os.path.exists(vitpose_annot_dir):
            annot_dir = vitpose_annot_dir
            print(f"Loading ViTPose annotations from {annot_dir}")
        else:
            annot_dir = original_annot_dir
            print(f"ViTPose annotations not found. Using original annotations from {annot_dir}")
            print(f"For better head crops, run: python preprocess/extract_vitpose_keypoints_rh.py --data_dir {RH_dir} --set_name {self.set_name}")

        self.annots = np.load(annot_dir, allow_pickle=True)['annots'][()]
        print(f'Loaded {sum(len(v) for v in self.annots.values())} persons from {self.set_name}.')
        # Only keep images that have annotations
        self.image_paths = [img_path for img_path in self.image_paths if os.path.basename(img_path) in self.annots]

    def build_person_list(self):
        # reorganize data to have one element per person per prompt
        self.persons = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            img_data = self.annots.get(img_name, [])
            if len(img_data) == 0:
                continue
            for p_idx, p_data in enumerate(img_data):
                bbox = p_data.get('bbox_wb', p_data['bbox'])  # use whole body bbox if available

                # Only use ViTPose body keypoints (COCO-17 format)
                # Skip persons without ViTPose keypoints
                keypoints = p_data.get('vitpose_body_keypoints', None)
                if keypoints is None:
                    print(f"Warning: Skipping person {p_idx} in {img_name} - no ViTPose keypoints found")
                    continue

                # add the person per prompt
                for pk, fn in zip(self.prompt_keys, self.feature_names):
                    self.persons.append({
                        "img_path": img_path,
                        "img_name": img_name,
                        "bbox": bbox,
                        "keypoints": keypoints,
                        "person_idx": p_idx,
                        "prompt_key": pk,
                        "feat_name": fn,
                    })

    def get_head_keypoints(self, keypoints):
        """
        Extract head keypoints from COCO-17 body keypoints.
        COCO-17 format: Nose(0), L_Eye(1), R_Eye(2), L_Ear(3), R_Ear(4),
                        L_Shoulder(5), R_Shoulder(6), ...

        We only use face keypoints: Nose, Eyes, Ears (indices 0-4)
        This matches the approach in ImageHeadCropDatasetPerson.
        """
        if keypoints is None:
            return None

        # Convert list to numpy array and reshape to (N, 3)
        kp_array = np.array(keypoints).reshape(-1, 3)
        num_kps = kp_array.shape[0]

        if num_kps >= 17:
            # COCO-17 body keypoints - extract face keypoints only
            # Indices: 0=Nose, 1=L_Eye, 2=R_Eye, 3=L_Ear, 4=R_Ear
            return kp_array[[0, 1, 2, 3, 4]]
        else:
            raise ValueError(f"Expected COCO-17 keypoints (17+), got {num_kps} keypoints. "
                           f"Please run ViTPose extraction first: "
                           f"python -m preprocess.extract_vitpose_keypoints_rh")

    def get_head_bbox(self, head_keypoints, x_pad_factor=0.3, top_pad_factor=2.0, bottom_pad_factor=0.5, bbox=None, valid_thresh=0.3, min_head_ratio=0.25):
        """
        Calculates a head bbox from face keypoints (COCO-17).
        Uses the same approach as ImageHeadCropDatasetPerson:
        - Get tight bbox from visible face keypoints
        - Apply padding: 30% horizontal, 200% top, 50% bottom
        """
        if head_keypoints is None or len(head_keypoints) == 0:
            return None

        def is_valid(kp):
            return kp is not None and kp[0] >= 0 and kp[1] >= 0 and kp[2] > valid_thresh

        # Get all visible face keypoints
        visible_kpts = np.array([kp for kp in head_keypoints if is_valid(kp)], dtype=np.float32)

        if visible_kpts.shape[0] == 0:
            return None

        # Get the tight bounding box of the face using all visible keypoints
        x_min, y_min, _ = np.min(visible_kpts, axis=0)
        x_max, y_max, _ = np.max(visible_kpts, axis=0)

        # Calculate face dimensions from the keypoints box
        face_width = x_max - x_min
        face_height = y_max - y_min

        # Faces are roughly square - use max dimension to handle back-of-head cases
        # where keypoints cluster horizontally
        face_size = max(face_width, face_height)

        # Only apply minimum when keypoints cluster tightly (back-of-head cases)
        # Don't override when face_size is already reasonable
        if bbox is not None and face_size < 20:
            person_height = bbox[3] - bbox[1]
            min_face_size = person_height * min_head_ratio
            face_size = max(face_size, min_face_size)

        # Apply padding based on the face size
        head_x1 = x_min - (face_size * x_pad_factor)
        head_y1 = y_min - (face_size * top_pad_factor)
        head_x2 = x_max + (face_size * x_pad_factor)
        head_y2 = y_max + (face_size * bottom_pad_factor)

        final_box = np.array([head_x1, head_y1, head_x2, head_y2]).astype(int)

        # Use the person's bounding box for clipping
        if bbox is not None:
            px1, py1, px2, py2 = bbox
            final_box = np.clip(
                [head_x1, head_y1, head_x2, head_y2],
                [px1, py1, px1, py1],  # Min bounds for head coordinates
                [px2, py2, px2, py2]   # Max bounds for head coordinates
            ).astype(int)

        # check if the head bbox is empty
        if final_box[0] >= final_box[2] or final_box[1] >= final_box[3]:
            return None
        return final_box

    def crop_head(self, image, bbox, keypoints):
        """
        Crop the head from the image based on the bounding box and keypoints.
        """
        head_keypoints = self.get_head_keypoints(keypoints)
        head_bbox = self.get_head_bbox(head_keypoints, bbox=bbox)

        # Fallback to person bbox if head bbox is not valid
        if head_bbox is None:
            head_bbox = bbox

        return self.crop_person(image, head_bbox)

    def __getitem__(self, idx):
        sample = self.persons[idx]
        img_name = sample["img_name"]
        image = Image.open(sample["img_path"]).convert("RGB")
        bbox = sample['bbox']
        keypoints = sample.get('keypoints', None)
        person_img = self.crop_head(image, bbox, keypoints)
        if self.transform:
            person_img = self.transform(person_img)

        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=person_img)

        inputs = None
        if self.processor is not None:
            inputs = self.prepare_inputs([person_img], [prompt])

        return {
            "image": person_img,
            "bbox": bbox,
            "img_name": img_name,
            "person_idx": sample["person_idx"],
            "prompt": prompt,
            "feat_name": sample["feat_name"],
            "inputs": inputs
        }


class RelativeHumanPersonOverlay(RelativeHumanPerson):
    def __init__(self, data_dir, model_name, transform=None, set_name='test',
                 processor=None, prompt_keys=['basic'], message=None, vision_processor=None,
                 min_size=None):
        super().__init__(data_dir, model_name, transform, set_name, processor, prompt_keys, message, vision_processor, min_size)
        self.transform = self.build_transform_from_processor()
    
    def overlay_bbox(self, image, bbox, color=(255, 0, 0), thickness=5):
        """
        Overlay a bounding box on the image.
        """
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        return image
    
    def __getitem__(self, idx):
        sample = self.persons[idx]
        img_name = sample["img_name"]
        image = Image.open(sample["img_path"]).convert("RGB")
        bbox = sample['bbox']
        person_img = self.overlay_bbox(image, bbox)
        if self.transform:
            person_img = self.transform(person_img)
        
        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=person_img)

        inputs = None
        if self.processor is not None:
            inputs = self.prepare_inputs([person_img], [prompt])

        return {
            "image": person_img,
            "bbox": bbox,
            "img_name": img_name,
            "person_idx": sample["person_idx"],
            "prompt": prompt,
            "feat_name": sample["feat_name"],
            "inputs": inputs
        }

class RelativeHumanPersonGrounded(RelativeHumanPerson):
    """
    Only for qwen models that support grounded captioning.
    """
    def __init__(self, data_dir, model_name, transform=None, set_name='test',
                 processor=None, prompt_keys=['basic'], message=None, vision_processor=None,
                 min_size=None):
        super().__init__(data_dir, model_name, transform, set_name, processor, prompt_keys, message, vision_processor, min_size)
        self.transform = self.build_transform_from_processor()

    def update_prompt(self, prompt, message, image=None, bbox=None):
        if image is not None:
            base64_img = self.image_to_base64(image)
        if isinstance(message, dict):
            text_idx = [idx for idx, i in enumerate(message["content"]) if i["type"] == "text"][0]
            image_idx = [idx for idx, i in enumerate(message["content"]) if i["type"] == "image"][0]
            message["content"][text_idx]["text"] = message["content"][text_idx]["text"].replace("{{text}}", prompt)
            if "image" in message["content"][image_idx]:
                message["content"][image_idx]["image"] = message["content"][image_idx]["image"].replace("{{img_url}}", base64_img)
            if bbox is not None:
                message["content"][text_idx]["text"] = message["content"][text_idx]["text"].replace("{{bbox}}", bbox)
        else:
            message = message.replace("{{text}}", prompt)
            message = message.replace("{{img_url}}", base64_img)

        if self.processor.chat_template:
            message = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        return message
    
    def transform_bbox(self, bbox, img_h, img_w):
        """
        Transforms the bounding box to the format expected by the model and produces the text prompt.
        """
        if self.model_name.startswith("Qwen/Qwen2.5-VL"):
            trans_bbox = self.convert_to_qwen25vl_format(bbox, img_h, img_w)
            bbox_prompt = f"<|box_start|>({trans_bbox[0]}, {trans_bbox[1]}),({trans_bbox[2]}, {trans_bbox[3]})<|box_end|>"
            return bbox_prompt
        else:
            raise ValueError(f"Model {self.model_name} does not support bounding box transformation.")
    
        # This is the resize function of Qwen2.5-VL
    def smart_resize(
        self, height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
    ):
        """Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def convert_to_qwen25vl_format(self, bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
        # from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/tools/process_bbox.ipynb
        new_height, new_width = self.smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
        scale_w = new_width / orig_width
        scale_h = new_height / orig_height
        
        x1, y1, x2, y2 = bbox
        x1_new = round(x1 * scale_w)
        y1_new = round(y1 * scale_h)
        x2_new = round(x2 * scale_w)
        y2_new = round(y2 * scale_h)
        
        x1_new = max(0, min(x1_new, new_width - 1))
        y1_new = max(0, min(y1_new, new_height - 1))
        x2_new = max(0, min(x2_new, new_width - 1))
        y2_new = max(0, min(y2_new, new_height - 1))
        
        return [x1_new, y1_new, x2_new, y2_new]
    
    def __getitem__(self, idx):
        sample = self.persons[idx]
        img_name = sample["img_name"]
        image = Image.open(sample["img_path"]).convert("RGB")
        img_width, img_height = image.size
        bbox = sample['bbox']
        bbox_prompt = self.transform_bbox(bbox, img_height, img_width)
        if self.transform:
            image = self.transform(image)
        
        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=image, bbox=bbox_prompt)

        inputs = None
        if self.processor is not None:
            inputs = self.prepare_inputs([image], [prompt])

        return {
            "image": image,
            "bbox": bbox,
            "img_name": img_name,
            "person_idx": sample["person_idx"],
            "prompt": prompt,
            "feat_name": sample["feat_name"],
            "inputs": inputs
        }
