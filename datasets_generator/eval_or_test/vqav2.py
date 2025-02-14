# vqav2_test_annotation_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vqav2/v2_mscoco_train2014_annotations.json"
vqav2_test_question_json = "/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/vqav2/v2_OpenEnded_mscoco_test2015_questions.json"

path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_files_small_dataset/vqav2_5k_test.json'

NUM_TESTING_SET = 5000

prefix_path_img = 'coco/test2015/'

import json, random
from tqdm import tqdm

random.seed(42)

TEMPLATE = {
    "id": "{question_id}",
    "image": prefix_path_img + "{image_id}.jpg",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} Answer the question using a single word or phrase.'
        },
        {
            "from": "gpt",
            "value": "{answer}"
        }
    ]
}


# with open(vqav2_test_annotation_json, 'r') as file:
#     test_annotations = json.load(file)['annotations']

with open(vqav2_test_question_json, 'r') as file:
    test_questions = json.load(file)['questions']
assert NUM_TESTING_SET <= len(test_questions), "The size of randomly sampled set cannot be larger than the original one."
sampled_questions = random.sample(test_questions, NUM_TESTING_SET)

# new_test_annotations = {}
# for anno in test_annotations:
#     new_test_annotations[anno['question_id']] = anno

test_list = []

for q in tqdm(sampled_questions):
    question = q['question']
    image_id = q['image_id']
    question_id = q['question_id']
    
    # test_annotation = new_test_annotations.get(question_id)
    # assert image_id == test_annotation['image_id']
    # answers = test_annotation['answers']
    # multiple_choice_answer = test_annotation['multiple_choice_answer']

    sample_data = {
        'question_id': question_id,
        'question': question,
        'answer': "",
        'image_id': "COCO_test2015_" + str(image_id).zfill(12)
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

print(f'The testing set has totally {len(test_list)} samples.')
print('Preprocessing succeeds!')