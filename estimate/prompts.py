def get_prompt(prompt_key):
    """
    Get the prompt for the given key.
    """
    return prompts_dict.get(prompt_key, None)

# Note the prompt keys are expected to have the feature name as the prefix
prompts_dict = {
        "age_basic": "Estimate the age of the person in the image. Provide a number between 0 and 100.",
        "age_cat1" : "Look at the image and estimate the person's age. Choose one of the following categories: baby, toddler, kid, teenager, or adult. Respond with just the category.",
        "age_verbose": """
            Look at the cropped image of a person and estimate their age group. Choose exactly one of the following categories according to the age in years:
            baby (0), toddler (1 to 3), kid (4 to 12), teen (13 to 19), adult (20+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_verbose_largest": """
            Look at the cropped image of a person and estimate their age group. Choose exactly one of the following categories according to the age in years:
            baby (0), toddler (1 to 3), kid (4 to 12), teen (13 to 19), adult (20+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            If there are multiple people in the image, focus on the largest person.
            """,
        "age_agedef_verbose": """
            Look at the cropped image of a person and estimate their age group.
            Choose one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), kid (4 to 8), teen (8 to 16), adult (16+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_agedef_verbose_headoverlay": """
            Look at the cropped image of a person and estimate their age group. If there is more than one person, focus on the one in the red bounding box.
            Choose one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), kid (4 to 8), teen (8 to 16), adult (16+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_agedef_verbose_unsure": """
            Look at the cropped image of a person and estimate their age group.
            Choose one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), kid (4 to 8), teen (8 to 16), adult (16+), unknown (?).
            Only respond with the category if you are sure, otherwise respond with "unknown".
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_sft_agedef_verbose": """
            Look at the cropped image of a person and estimate their age group.
            Choose one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), child (4 to 8), teenager (8 to 16), adult (16+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_cat2": "Look at the image and estimate the person's age. Choose one of the following categories: baby, toddler, child, teenager, adult, senior. Respond with just the category.",
        "age_overlay": "Look at the image and estimate the age of the largest person in the red bounding box. Choose one of the following categories: baby, toddler, child, teenager, adult, senior. Respond with just the category.",
        "age_overlayagedef": "Look at the image and estimate the age of the largest person in the red bounding box. Choose one of the following categories according to the age: baby (0 to 1), toddler (2 to 3), child (4 to 8), teenager (8 to 16), adult (16+). Respond with just the category.",
        "age_overlayagedef_verbose": """
            Look at the image and estimate the age of the largest person in the red bounding box.
            Choose one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), kid (4 to 8), teen (8 to 16), adult (16+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_overlay_verbose": """
            Look at the image and estimate the age of the largest person in the red bounding box.
            Choose exactly one of the following categories according to the age in years: baby (0), toddler (1 to 3), kid (4 to 12), teen (13 to 19), adult (20+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_agedef_verbose_grounded": """
            What is the age of the person in the region {{bbox}}?
            Choose exactly one of the following categories according to the age in years: baby (0 to 1), toddler (2 to 3), kid (4 to 8), teen (8 to 16), adult (16+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_grounded": """
            What is the age of the person in the region {{bbox}}?
            Choose exactly one of the following categories according to the age in years: baby (0), toddler (1 to 3), kid (4 to 12), teen (13 to 19), adult (20+).
            Respond with only the category name.
            For example, if the person is a toddler, respond with "toddler".
            """,
        "age_sft": "After observing the source image, could you please answer the following: What is the approximate age group of the person in the image?",
        "age_sft_catnames": "After observing the source image, could you please answer the following: What is the approximate age group of the person in the image?Choose one of the following categories: baby, toddler, child, teenager, adult. Respond with just the category.",
        "gender_cat1": """Look at the image and estimate the person's gender. Choose one of the following categories: male, female, unknown. Respond with just the category.""",
        "gender_overlay": """Look at the image and estimate the gender of the largest person in the red bounding box. Choose one of the following categories: male, female, unknown. Respond with just the category.""",
        "gender_verbose": """
        Look at the cropped image of a person and estimate their gender. Choose exactly one of the following categories: male, female, unknown.
        Respond with only the category name.
        For example, if the person is female, respond with "female".
        """,
        "gender_verbose_headoverlay": """
        Look at the cropped image of a person and estimate their gender. Choose exactly one of the following categories: male, female, unknown.
        Respond with only the category name.
        For example, if the person is female, respond with "female".
        """,
        "gender_verbose_grounded": """
        What is the gender of the person in the region {{bbox}}.
        Choose exactly one of the following categories: male, female, unknown.
        Respond with only the category name.
        For example, if the person is female, respond with "female".
        """,
        "gender_verbose_overlay": """
        Look at the cropped image of a person and estimate their gender. If there is more than one person, focus on the one in the red bounding box.
        Choose exactly one of the following categories: male, female, unknown.
        Respond with only the category name.
        For example, if the person is female, respond with "female".
        """,
        "height_basic": "Estimate the height of the person in the image. Return a single number in centimeters",
        "height_v2": "How tall is the person in this image if they were standing up straight?. Return a single number in centimeters",
        "mass_basic": "How heavy is this person? Return a single number in kg",
        "hips_basic": "What does the circunference of this person's hips measure? Return a single number in centimeters",
        "waist_basic": "What does the circunference of this person's waist measure? Return a single number in centimeters",
        "chest_basic": "What does the circunference of this person's chest measure? Return a single number in centimeters",
        "shape_categories": """Classify the person's body shape. Choose exactly one category:
        - slim: Very low body fat, narrow frame.
        - average: Balanced build, no extreme features.
        - overweight: Noticeable body fat, rounded midsection.
        - muscular: High muscle mass, low body fat.
        Respond with only the category name.
        For example, if the person is slim, respond with "slim"."""
    }