import re

AGE_MAPPING = {'baby': 0.0, 'toddler': 0.1, 'kid': 0.330, 'child': 0.330, 'teenager': 0.500, 'teen': 0.500, 'adult': 0.660, 'elder': 0.999, "senior": 0.999}
GENDER_MAPPING = {'male': 0, 'female': 1, 'neutral': 0.5, 'unknown': 0.5}

ANNY_MAPPING = {
    'age': AGE_MAPPING,
    'gender': GENDER_MAPPING,
}

def map_prediction(pred, map_dict=None, default=-1):
    # handle predictions that don't exactly match the keys in the mapping dict
    pred = pred.lower().strip()
    if map_dict is None:
        return pred
    for key in map_dict:
        if re.search(rf'\b{key}\b', pred):
            return map_dict[key]
    return default

def attributes2shape(person_attributes, model_type='anny', default=-1.0):
    if model_type != 'anny':
        return None
    age = default
    age_name = person_attributes.get('age', None)
    if age_name is not None:
        age = map_prediction(age_name, ANNY_MAPPING['age'], default=default)

    gender = default
    gender_name = person_attributes.get('gender', None)
    if gender_name is not None:
        gender = map_prediction(gender_name, ANNY_MAPPING['gender'], default=default)

    return {'age': age, 'gender': gender}
