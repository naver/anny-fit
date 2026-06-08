import copy
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from torch.utils.data import Dataset

from estimate.prompts import get_prompt


class ImageCropDatasetPerson(Dataset):
    def __init__(self, processed_file, model_name, processor, prompt_keys=['basic'], message=None):
        loaded_npz = np.load(processed_file, allow_pickle=True)
        self.det_data = {key: loaded_npz[key] for key in loaded_npz.files}
        self.model_name = model_name
        self.image_path = self.det_data["img_path"].item()
        self.img_name = self.det_data["imgname"].item()
        self.processor = processor
        self.prompt_keys = prompt_keys
        self.feature_names = [pk.split("_")[0] for pk in prompt_keys]
        self.message = message
        self.build_person_list()

    def build_person_list(self):
        self.persons = []
        for p_idx, bbox in zip(self.det_data['person_ids'], self.det_data['bboxes']):
            for pk, fn in zip(self.prompt_keys, self.feature_names):
                self.persons.append({
                    "img_path": self.image_path,
                    "img_name": self.img_name,
                    "bbox": bbox,
                    "person_idx": p_idx,
                    "prompt_key": pk,
                    "feat_name": fn,
                })

    def organize_predictions(self, all_preds):
        organized = {}
        for person in self.persons:
            img_name = person["img_name"]
            p_idx = person["person_idx"]
            fn = person["feat_name"]
            if str(p_idx) not in organized:
                organized[str(p_idx)] = {}
            organized[str(p_idx)][fn] = all_preds.get((img_name, p_idx, fn), '')
        return organized

    def crop_person(self, image, bbox):
        x1, y1, x2, y2 = bbox  # (x1, y1): top-left, (x2, y2): bottom-right
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image.crop((x1, y1, x2, y2))

    def image_to_base64(self, image):
        img_format = image.format or "JPEG"
        buffer = BytesIO()
        image.save(buffer, format=img_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    def update_prompt(self, prompt, template, image=None):
        message = copy.deepcopy(template)  # necessary for multiple prompts
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
        images = [item["person_img"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        inputs = self.prepare_inputs(images, prompts)
        return {
            "inputs": inputs,
            "img_names": [item["img_name"] for item in batch],
            "person_idxs": [item["person_idx"] for item in batch],
            "feat_names": [item["feat_name"] for item in batch],
        }

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        sample = self.persons[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        person_img = self.crop_person(image, sample["bbox"])
        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=person_img)
        return {
            "person_img": person_img,
            "img_name": sample["img_name"],
            "person_idx": sample["person_idx"],
            "feat_name": sample["feat_name"],
            "prompt": prompt,
        }


class ImageMaskedCropDatasetPerson(ImageCropDatasetPerson):
    """Crop by bbox but mask out pixels outside the person's segmentation mask to white."""

    def __getitem__(self, idx):
        sample = self.persons[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        img_np = np.array(image)

        # Apply mask: set pixels outside this person's mask to white
        mask = sample["mask"]  # (H, W) binary
        img_np[mask < 0.5] = 255

        masked_image = Image.fromarray(img_np)
        person_img = self.crop_person(masked_image, sample["bbox"])
        prompt = get_prompt(sample["prompt_key"])
        if self.message is not None:
            msg = copy.deepcopy(self.message)
            prompt = self.update_prompt(prompt, msg, image=person_img)
        return {
            "person_img": person_img,
            "img_name": sample["img_name"],
            "person_idx": sample["person_idx"],
            "feat_name": sample["feat_name"],
            "prompt": prompt,
        }

    def build_person_list(self):
        self.persons = []
        masks = self.det_data.get('masks', None)
        for p_idx, bbox in zip(self.det_data['person_ids'], self.det_data['bboxes']):
            mask = masks[p_idx] if masks is not None else None
            for pk, fn in zip(self.prompt_keys, self.feature_names):
                self.persons.append({
                    "img_path": self.image_path,
                    "img_name": self.img_name,
                    "bbox": bbox,
                    "mask": mask,
                    "person_idx": p_idx,
                    "prompt_key": pk,
                    "feat_name": fn,
                })


class ImageHeadCropDatasetPerson(ImageCropDatasetPerson):
    def __init__(self, processed_file, model_name, processor, prompt_keys=['basic'], message=None):
        super().__init__(processed_file, model_name, processor, prompt_keys, message)

    def build_person_list(self):
        self.persons = []
        for p_idx, bbox in zip(self.det_data['person_ids'], self.det_data['bboxes']):
            person_keypoints = self.det_data['all_keypoints'][p_idx]
            head_keypoints = self.get_head_keypoints(person_keypoints)
            head_bbox = self.get_head_bbox(head_keypoints, bbox=bbox)
            if head_bbox is None:
                head_bbox = bbox  # fallback to person bbox if head bbox is not valid
            for pk, fn in zip(self.prompt_keys, self.feature_names):
                self.persons.append({
                    "img_path": self.image_path,
                    "img_name": self.img_name,
                    "bbox": head_bbox,
                    "person_idx": p_idx,
                    "prompt_key": pk,
                    "feat_name": fn,
                })

    def get_head_keypoints(self, keypoints):
        """
        Assumes points are in coco whole body format
        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints
        """
        indices = np.concatenate((np.arange(0, 5), np.arange(23, 91)))
        return keypoints[indices]

    def get_head_bbox(self, head_keypoints, x_pad_factor=0.3, top_pad_factor=2.0, bottom_pad_factor=0.5, bbox=None, valid_thresh=0.3):
        """
        Calculates a head bbox from head keypoints
        """
        def is_valid(kp):
            return kp is not None and kp[0] >= 0 and kp[1] >= 0 and kp[2] > valid_thresh

        visible_kpts = np.array([kp for kp in head_keypoints if is_valid(kp)], dtype=np.float32)

        if visible_kpts.shape[0] == 0:
            return None

        x_min, y_min, _ = np.min(visible_kpts, axis=0)
        x_max, y_max, _ = np.max(visible_kpts, axis=0)

        face_width = x_max - x_min
        face_height = y_max - y_min

        head_x1 = x_min - (face_width * x_pad_factor)
        head_y1 = y_min - (face_height * top_pad_factor)
        head_x2 = x_max + (face_width * x_pad_factor)
        head_y2 = y_max + (face_height * bottom_pad_factor)

        final_box = np.array([head_x1, head_y1, head_x2, head_y2]).astype(int)

        if bbox is not None:
            px1, py1, px2, py2 = bbox
            final_box = np.clip(
                [head_x1, head_y1, head_x2, head_y2],
                [px1, py1, px1, py1],  # min bounds for head coordinates
                [px2, py2, px2, py2]   # max bounds for head coordinates
            ).astype(int)

        if final_box[0] >= final_box[2] or final_box[1] >= final_box[3]:
            return None
        return final_box
