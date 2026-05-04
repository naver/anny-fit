import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}
DEMO = {
    "annotation_path": "single_images.json",
    "data_path": "",
}
AGE_RH = {
    "annotation_path": "data/age_rh.json",
    "data_path": "",
}
WIKI = {
    "annotation_path": "data/wiki.json",
    "data_path": "",}

IMDB = {
    "annotation_path": "data/imdb.json",
    "data_path": "",
}

UTKFACE_TRAIN = {
    "annotation_path": "data/utkface_train.json",
    "data_path": "",
}

UTKFACE_TRAIN_RESAMPLED = {
    "annotation_path": "data/utkface_resampled_train.json",
    "data_path": "",
}

UTKFACE_TRAIN_RESAMPLED_NORM = {
    "annotation_path": "data/utkface_resampled_train_normage.json",
    "data_path": "",
}

UTKFACE_TRAIN_RESAMPLED_NUM = {
    "annotation_path": "data/utkface_resampled_train_agenum.json",
    "data_path": "",
}

UTKFACE_TRAIN_RESAMPLED_ALL = {
    "annotation_path": "data/utkface_resampled_train_all.json",
    "data_path": "",
}

ALL_AGES_FACES_TRAIN = {
    "annotation_path": "data/all_ages_faces_train.json",
    "data_path": "",
}

ALL_AGES_FACES_TRAIN_RESAMPLED = {
    "annotation_path": "data/all_ages_faces_resampled_train.json",
    "data_path": "",
}

ALL_AGES_FACES_TRAIN_RESAMPLED_NORM = {
    "annotation_path": "data/all_ages_faces_resampled_train_normage.json",
    "data_path": "",
}

ALL_AGES_FACES_TRAIN_RESAMPLED_NUM = {
    "annotation_path": "data/all_ages_faces_resampled_train_agenum.json",
    "data_path": "",
}

ALL_AGES_FACES_TRAIN_RESAMPLED_ALL = {
    "annotation_path": "data/all_ages_faces_resampled_train_all.json",
    "data_path": "",
}

ALL_AGES_FACES_VAL = {
    "annotation_path": "data/all_ages_faces_resampled_val.json",
    "data_path": "",
}

ALL_AGES_FACES_VAL_RESAMPLED = {
    "annotation_path": "data/all_ages_faces_resampled_val.json",
    "data_path": "",
}

ALL_AGES_FACES_VAL_RESAMPLED_NORM = {
    "annotation_path": "data/all_ages_faces_resampled_val_normage.json",
    "data_path": "",
}

ALL_AGES_FACES_VAL_RESAMPLED_NUM = {
    "annotation_path": "data/all_ages_faces_resampled_val_agenum.json",
    "data_path": "",
}

ALL_AGES_FACES_VAL_RESAMPLED_ALL = {
    "annotation_path": "data/all_ages_faces_resampled_val_all.json",
    "data_path": "",
}

data_dict = {
    "demo": DEMO,
    "age_RH": AGE_RH,
    "wiki": WIKI,
    "utkface_train": UTKFACE_TRAIN,
    "all_ages_faces_train": ALL_AGES_FACES_TRAIN,
    "all_ages_faces_val": ALL_AGES_FACES_VAL,

    "utkface_resampled_train": UTKFACE_TRAIN_RESAMPLED,
    "all_ages_faces_resampled_train": ALL_AGES_FACES_TRAIN_RESAMPLED,
    "all_ages_faces_resampled_val": ALL_AGES_FACES_VAL_RESAMPLED,

    "utkface_resampled_train_normage": UTKFACE_TRAIN_RESAMPLED_NORM,
    "all_ages_faces_resampled_train_normage": ALL_AGES_FACES_TRAIN_RESAMPLED_NORM,
    "all_ages_faces_resampled_val_normage": ALL_AGES_FACES_VAL_RESAMPLED_NORM,

    "utkface_resampled_train_numage": UTKFACE_TRAIN_RESAMPLED_NUM,
    "all_ages_faces_resampled_train_numage": ALL_AGES_FACES_TRAIN_RESAMPLED_NUM,
    "all_ages_faces_resampled_val_numage": ALL_AGES_FACES_VAL_RESAMPLED_NUM,

    "imdb": IMDB,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["utkface_train"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
