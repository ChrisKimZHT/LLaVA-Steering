# source_paths
iconqa_data_root_path = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/raw/iconQA/iconqa_data'

# arguments for saving
dir2save_img = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval'
path2save_json = '/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_files_small_dataset/iconQA_blank_test.json'

prefix_path_img = "iconQA_blank/test/"

# templates
TEMPLATE = {
    "id": "{sample_id}",
    "image": prefix_path_img + "{sample_id}.png",
    "conversations": [
        {
            "from": "human",
            "value": '<image>\n{question} Fill in the blanks in (_) or answer this question'
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
from tqdm import tqdm

pid_splits_json = os.path.join(iconqa_data_root_path, 'pid_splits.json')
train_root = os.path.join(iconqa_data_root_path, 'iconqa/train')
val_root = os.path.join(iconqa_data_root_path, 'iconqa/val')
test_root = os.path.join(iconqa_data_root_path, 'iconqa/test')

test_blank_root = os.path.join(test_root, 'fill_in_blank')

test_dir_names = os.listdir(test_blank_root)

test_list = []

os.makedirs(os.path.join(dir2save_img, prefix_path_img), exist_ok=True)
os.makedirs(os.path.dirname(path2save_json), exist_ok=True)

for dir_name in tqdm(test_dir_names):
    sample_path = os.path.join(test_blank_root, dir_name)
    sample_json = os.path.join(sample_path, 'data.json')
    img_path = os.path.join(sample_path, 'image.png')

    with open(sample_json, 'r') as file:
        sample_data = json.load(file)
    question = sample_data['question']
    answer = sample_data['answer']

    sample_data = {
        'sample_id': dir_name,
        'question': question,
        'answer': answer,
    }

    formatted_sample = {
        key: (value.format(**sample_data)
            if isinstance(value, str)
            else [{k: v.format(**sample_data)
                for k, v in item.items()} for item in value])
        for key, value in TEMPLATE.items()
    }

    test_list.append(formatted_sample)

    shutil.copy2(img_path,
                 os.path.join(dir2save_img, prefix_path_img+dir_name+".png"))


with open(path2save_json, 'w') as outfile:
    json.dump(test_list, outfile, indent=4)

print(f"Test set has {len(test_list)} samples!")
print('Preprocessing succeeds!')

