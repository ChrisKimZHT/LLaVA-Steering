vizwiz_test_img_dir = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vizwiz/val"
vizwiz_test_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vizwiz/val.json"

dir2save_img = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval'
path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_files_small_dataset/vizwiz.json'

prefix_path_img = 'vizwiz/val/'

# templates
TEMPLATE = {
    "id": "{sample_id}",
    "image": prefix_path_img + "{image}",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} When the information is insufficient, respond with "Unanswerable". Answer the question using a single word or phrase.'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


import shutil
import os
import json
import random
import re
from tqdm import tqdm

random.seed(42)

with open(vizwiz_test_json, 'r') as file:
    test_data = json.load(file)
    
test_list = []

os.makedirs(os.path.join(dir2save_img, prefix_path_img), exist_ok=True)
os.makedirs(os.path.dirname(path2save_json), exist_ok=True)


for sample_data in tqdm(test_data):
    img_name = sample_data['image']
    question = sample_data['question']
    answers = sample_data['answers']
    answerable = sample_data['answerable']
    answer_type = sample_data['answer_type']

    if answerable == 0:
        answer = 'Unanswerable'
    else:
        filtered_answers = [ans for ans in answers if ans["answer_confidence"] in ["yes", "maybe"] and ans['answer'] != 'unanswerable']
        weights = [3 if ans["answer_confidence"] == "yes" else 1 for ans in filtered_answers]
        answer = random.choices(filtered_answers, weights=weights, k=1)[0]['answer']

    sample_id = re.findall(r'\d+', img_name)[0]

    sample_data = {
        'sample_id': sample_id,
        'question': question,
        'answer': answer,
        'image': img_name
    }

    img_path = os.path.join(vizwiz_test_img_dir, img_name)

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    test_list.append(formatted_sample)

    shutil.copy2(img_path,
                 os.path.join(dir2save_img, prefix_path_img+img_name))


with open(path2save_json, 'w') as outfile:
    json.dump(test_list, outfile, indent=4)

print(f"The size of val set of vizwiz is {len(test_list)}!")
print('Preprocessing succeeds!')

