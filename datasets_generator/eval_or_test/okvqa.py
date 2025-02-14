okvqa_test_question_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/okvqa/OpenEnded_mscoco_val2014_questions.json"
okvqa_test_annotation_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/okvqa/mscoco_val2014_annotations.json"

path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_files_small_dataset/okvqa.json'

import json, random
from tqdm import tqdm

random.seed(42)

prefix_path_img = 'coco/val2014/'

TEMPLATE = {
    "id": "{question_id}",
    "image": prefix_path_img + "{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question}'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


with open(okvqa_test_annotation_json, 'r') as file:
    test_annotations = json.load(file)['annotations']

with open(okvqa_test_question_json, 'r') as file:
    test_questions = json.load(file)['questions']


new_test_annotations = {}
for anno in test_annotations:
    new_test_annotations[anno['question_id']] = anno

test_list = []

for q in tqdm(test_questions):
    question = q['question']
    image_id = q['image_id']
    question_id = q['question_id']
    
    test_annotation = new_test_annotations.get(question_id)
    assert image_id == test_annotation['image_id']
    answers = test_annotation['answers']

    answer = random.choices(answers, k=1)[0]['answer']

    sample_data = {
        'question_id': question_id,
        'question': question,
        'answer': answer,
        'image_id': 'COCO_val2014_' + str(image_id).zfill(12)
    }

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    test_list.append(formatted_sample)

with open(path2save_json, 'w') as outfile:
    json.dump(test_list, outfile, indent=4)

print(f"The size of testset is {len(test_list)}")
print('Preprocessing succeeds!')
