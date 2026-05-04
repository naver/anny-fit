import re

def parse_prediction(output: str, feat_name="age", is_numeric=False): #, schema=PersonAttributes):
    # TODO: use schema and structured outputs
    # lowercase, remove excess whitespace and punctuation
    output = output.lower()
    if "assistant" in output:
        # sometimes the output includes the question again
        output = output.split("assistant")[-1]
    
    cleaned = re.sub(r'[^\w\s]', '', output).strip()
    # If cleaned output is just a single word, return it directly
    if len(cleaned.split()) == 1:
        return cleaned
    # try to extract value after colon, or fallback to last word
    if is_numeric:
        match = re.search(r'\d+\.?\d*', cleaned)
        group_num = 0
    else:
        match = re.search(rf"{feat_name}\s*[:\-]?\s*(\w+)", cleaned)
        group_num = 1

    value = match.group(group_num) if match else cleaned.split()[-1]
    return value

def compute_feat_metrics(p, tokenizer, feat_name="age", is_numeric=False):
    decoded = tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    parsed_preds = [parse_prediction(pred, feat_name, is_numeric) for pred in decoded]
    import ipdb; ipdb.set_trace()