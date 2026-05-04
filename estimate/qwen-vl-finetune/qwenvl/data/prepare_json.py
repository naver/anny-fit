import os
import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
import numpy as np

# set seed for reproducibility
random.seed(2025)
torch.manual_seed(2025)



# from https://github.com/WisconsinAIVision/ViP-LLaVA/blob/main/llava/visual_prompt_organizer.py
question_prefixes = [
    'Based on the provided source image, please answer this question: ',
    'In the context of the source image, can you answer: ',
    'With reference to the source image, please respond to the following query: ',
    "Considering the source image, what's your answer to: ",
    'Please provide an answer for the subsequent question, keeping the source image in mind: ',
    'Taking into account the source image, please answer: ',
    'After observing the source image, could you please answer the following: ',
    'Upon examining the source image, what would your answer be to: ',
    'Using the source image as a reference, please respond to: ',
    'In light of the source image, could you please answer: '
]

age_norm_questions = [
    "What is the normalized age of the person in the image?",
    "How old is the person in the image in normalized years?",
    "Can you estimate the age of the person in the image in normalized years?",
    "What is the approximate normalized age of the person in the image?",
    "Can you tell me the normalized age of the person in the image?",
]

age_name_questions = [
    "What is the age of the person in the image?",
    "How old is the person in the image?",
    "Can you estimate the age of the person in the image?",
    "What age group does the person in the image belong to?",
    "What is the approximate age group of the person in the image?",
    "Can you tell me the age class of the person in the image?",
]

age_num_questions = [
    "What is the age in years of the person in the image?",
    "How old is the person in the image? Give the age in years.",
    "Can you estimate the age of the person in the image in years?",
    "What is the approximate age in years of the person in the image?",
    "Can you tell me the age in years of the person in the image?",
]

gender_questions = [
    "What is the gender of the person in the image?",
    "Is the person in the image male, female or unknown?",
    "Can you identify the gender of the person in the image?",
    "What gender does the person in the image appear to be?",
    "How would you classify the gender of the person in the image?",
    "Can you tell me the gender of the person in the image?",
]

questions_dict = {'age_name': age_name_questions,
                  'age_norm': age_norm_questions,
                  'age_num': age_num_questions,
                  'gender': gender_questions}

# max is not inclusive
AGE_MAPPING = {
    'baby': {"min": 0, "max": 2},
    'toddler': {"min": 2, "max": 4},
    'child': {"min": 4, "max": 13},
    'teenager': {"min": 13, "max": 16},
    'adult': {"min": 16, "max": 70},
    'senior': {"min": 70, "max": 150},
}

class ShapeDataset(Dataset):
    def __init__(self, save_dir, data_path, question_types, set_name="train", dataset_name="shape", valid_question_types=["age"]):
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, f"{dataset_name}_resampled_{set_name}_agenum.json")

        self.data_path = data_path
        self.init_img_dir()
        self.set_name = set_name
        self.question_types = question_types
        self.check_question_types(valid_question_types)

        os.makedirs(self.save_dir, exist_ok=True)
        self.img_names = self.get_set_names()
        self.answers_dict = self.load_answers()
        self.answers_dict = self.resample_age_answers(self.answers_dict)
        self.instance_info = self.prepare_instance_info()

    def init_img_dir(self):
        self.img_dir = os.path.join(self.data_path, "images")

    def check_question_types(self, valid_question_types):
        valid_types = set(valid_question_types)
        type_set = set(self.question_types)
        if not type_set.issubset(valid_types):
            print(f"Invalid question types: {type_set - valid_types}. Updating question types to {valid_types}")
            self.question_types = valid_question_types

    def get_set_names(self):
        """
        Get the list of image names in the dataset split
        Assumes that all the images are training images
        """
        print(f"Assuming all image are training images")
        img_names = []
        if self.set_name == "train":
            img_names = os.listdir(self.img_dir)
        return img_names
        
    def load_answers(self, resample_age=True):
        """
        Read annotations to format answers into dict by question type.
        answers_dict[image_name][question_type] = answer
        """
        answers_dict = {}
        # Load annotations and populate answers_dict
        return answers_dict

    def resample_age_answers(self, answers_dict, target_class='child', factor=3):
        # balance the age samples
        ages = [v["age_name"] for k, v in answers_dict.items()]
        age_counts = {age_class: ages.count(age_class) for age_class in AGE_MAPPING.keys()}
        print(f"Original counts: {age_counts}")
        # determine the target size of the largest child class
        if self.set_name != "train":
            factor = 1
        target_size = age_counts[target_class] * factor
        majority_class = max(age_counts, key=age_counts.get)

        if age_counts[majority_class] > target_size:
            # group all samples by their class
            items_by_class = {age_class: [] for age_class in age_counts}
            for key, value in answers_dict.items():
                items_by_class[value["age_name"]].append((key, value))

            # downsample the majority class
            majority_items = items_by_class[majority_class]
            if len(majority_items) > target_size:
                items_by_class[majority_class] = random.sample(majority_items, target_size)

            # rebuild the dictionary with the balanced data
            resampled_answers_dict = {}
            for items in items_by_class.values():
                for key, value in items:
                    resampled_answers_dict[key] = value
                    
            # overwrite the old dictionary with the resampled one
            ages = [v["age_name"] for k, v in resampled_answers_dict.items()]
            age_counts = {age_class: ages.count(age_class) for age_class in AGE_MAPPING.keys()}
            print(f"Resampled {self.set_name} set to balance age classes: {age_counts}")
            answers_dict = resampled_answers_dict
        return answers_dict


    def prepare_instance_info(self):
        """
        Assumes one person per image.
        Returns data formated for preparing the JSON file.
        """
        instance_info = []
        for img_name in self.img_names:
            img_path = os.path.join(self.img_dir, img_name)
            question_answers = self.answers_dict.get(img_name, {})
            if not question_answers:
                continue
            for question_type in self.question_types:
                type_answer = question_answers.get(question_type)
                if type_answer is None:
                    # skip invalid answer to question
                    continue
                instance_info.append({
                    "image": img_path,
                    "question": question_type,
                    "answer": type_answer,
                })
        return instance_info
    
    def normalize_age(self, real_age):
        """
        Normalizes a real age using piecewise linear interpolation
        based on the defined AGE_ANCHORS.
        """
        AGE_ANCHORS = {0: 0.0, 1.0: 0.05, 4.0: 0.215, 11.0: 0.415, 16.0: 0.66, 18.0: 0.77, 64.0: 0.83, 110.0: 1.0}
        # Extract the real ages (x-points) and normalized ages (y-points)
        real_age_points = list(AGE_ANCHORS.keys())
        norm_age_points = list(AGE_ANCHORS.values())
        
        return np.interp(real_age, real_age_points, norm_age_points)
    
    def age_to_num(self, age):
        is_valid = age.isdigit() and int(age) >= 0
        if is_valid:
            return str(int(age))
        else:
            return None
    
    def age_to_normalizedage(self, age):
        is_valid = age.isdigit() and int(age) >= 0
        if is_valid:
            norm_age = self.normalize_age(age)
            return f"{norm_age:.2f}"
        else:
            return None
        
    def age_to_name(self, age):
        for age_group, age_range in AGE_MAPPING.items():
            if age_range["min"] <= age < age_range["max"]:
                return age_group
        return None

        
    def __len__(self):
        return len(self.instance_info)


    def __getitem__(self, idx):
        instance = self.instance_info[idx]
        img_path = instance["image"]
        question_type = instance["question"]
        answer = instance["answer"]
        # Prepare the JSON entry for a single image
        json_entry = self.json_single_image(img_path, question_type, answer)
        
        return json_entry
    
    def json_single_image(self, img_path, question_type, answer):
        """
        Prepare a single image JSON entry. Use random prefixes to get diversity of prompts.
        """
        prefix_idx = torch.randint(0, len(question_prefixes), (1,)).item()
        question_bank = questions_dict[question_type]
        question_idx = torch.randint(0, len(question_bank), (1,)).item()
        full_question = question_prefixes[prefix_idx] + question_bank[question_idx]
        
        return {
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + full_question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
                ]
            }
    
    def save_to_json(self, json_data):
        """
        Save the dataset to a JSON file.
        """
        with open(self.save_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Dataset saved to {self.save_path}")
        print(f"Number of questions: {len(json_data)}")

class UTKFace(ShapeDataset):
    def __init__(self, *args):
        valid_question_types = ["age_num", "age_norm", "age_name", "gender"]
        super().__init__(*args, dataset_name="utkface", valid_question_types=valid_question_types)
    
    def gender_to_name(self, gender):
        gender_mapping = {'0': "male", '1': "female", '': "unknown"}
        return gender_mapping.get(gender, None)
    
    def race_to_name(self, race):
        race_mapping = {'0': "white", '1': "black", '2': "asian", '3': "indian", '4': "other"}
        return race_mapping.get(race, None)

    def load_answers(self):
        """
        Read annotations to format answers into dict by question type.
        answers_dict[image_name][question_type] = answer
        """
        answers_dict = {}
        for img_name in self.img_names:
            age, gender, race, _ = img_name.split('_')
            # convert values to categories
            age_num = self.age_to_num(age)
            age_name = self.age_to_name(int(age))
            age_norm = self.age_to_normalizedage(age)
            gender_name = self.gender_to_name(gender)
            race_name = self.race_to_name(race)

            answers_dict[img_name] = {
                "age_name": age_name,
                "age_norm": age_norm,
                "age_num": age_num,
                "gender": gender_name,
                "race": race_name
            }

        return answers_dict


class AllAgesFaces(ShapeDataset):
    def __init__(self, *args):
        valid_question_types = ["age_num", "age_norm", "age_name", "gender"]
        super().__init__(*args, dataset_name="all_ages_faces", valid_question_types=valid_question_types)

    def init_img_dir(self):
        self.img_dir = os.path.join(self.data_path, "original_images")
    
    def get_set_names(self):
        """
        Get the list of image names in the dataset split
        Assumes that all the images are training images
        """
        img_names = []
        if self.set_name == "train" or self.set_name == "val":
            file_name = os.path.join(self.data_path, "image_sets", f"{self.set_name}.txt")
            # read image names from the file: img_name, gender
            with open(file_name, 'r') as f:
                img_names = []
                self.genders_dict = {}
                for line in f.readlines():
                    img_name, gender = line.strip().split(' ')
                    img_names.append(img_name)
                    self.genders_dict[img_name] = gender
        return img_names
    
    def gender_to_name(self, gender):
        # https://github.com/JingchunCheng/All-Age-Faces-Dataset?tab=readme-ov-file
        gender_mapping = {'0': "female", '1': "male", '': "unknown"}
        return gender_mapping.get(gender, None)

    def load_answers(self):
        """
        Read annotations to format answers into dict by question type.
        answers_dict[image_name][question_type] = answer
        """
        answers_dict = {}
        for img_name in self.img_names:
            person_id, age = img_name.split('.')[0].split('A')
            # convert values to categories
            age_num = self.age_to_num(age)
            age_name = self.age_to_name(int(age))
            age_norm = self.age_to_normalizedage(age)
            gender_name = self.gender_to_name(self.genders_dict.get(img_name, ''))

            answers_dict[img_name] = {
                "age_num": age_num,
                "age_norm": age_norm,
                "age_name": age_name,
                "gender": gender_name,
            }

        return answers_dict


def prepare_json(save_dir: str, data_path: str, question_types: list, dataset_name: str, set_name: str = "train"):
    """
    Prepare the JSON file for the dataset.
    """
    if dataset_name == "utkface":
        dataloader = UTKFace(save_dir, data_path, question_types, set_name)
    elif dataset_name == "all_ages_faces":
        dataloader = AllAgesFaces(save_dir, data_path, question_types, set_name)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported for JSON preparation.")

    json_data = [dataloader[i] for i in tqdm(range(len(dataloader)))]
    dataloader.save_to_json(json_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare JSON dataset for VLM training.")
    parser.add_argument("--save_dir", type=str, help="Directory to save the JSON file", default="data")
    parser.add_argument("--data_path", type=str, help="Path to the dataset directory", default="data/All-Ages-Faces")
    parser.add_argument("--question_types", type=str, nargs='+', default=["age_num", "age_norm", "age_name", "gender"])
    parser.add_argument("--set_name", type=str, default="train", help="Dataset split name", choices=["train", "val", "test"])
    parser.add_argument("--dataset_name", type=str, default="all_ages_faces", help="Name of the dataset", choices=["wiki", "imdb", "utkface", "all_ages_faces"])
    args = parser.parse_args()
    prepare_json(args.save_dir, args.data_path, args.question_types, args.dataset_name, set_name=args.set_name)