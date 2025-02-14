import argparse
import json
import os
import re
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-folder', type=str)
    parser.add_argument('--answers-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-result', type=str)
    return parser.parse_args()


def generate_questions_imgs(test_folder):
    data_list = []
    test_folder = os.path.expanduser(test_folder)
    ids = os.listdir(test_folder)
    for id in ids:
        img_abs_path = os.path.join(os.path.join(test_folder, id), 'image.png')
        json_path = os.path.join(os.path.join(test_folder, id), 'data.json')
        data = json.load(open(json_path, "r"))
        data_list.append({
            'id': id,
            'question': data['question'],
            'answer': data['answer'],
            'image': img_abs_path
        })
    return data_list


if __name__ == "__main__":


    args = get_args()

    data_list = generate_questions_imgs(args.test_folder)
    GTs = {data['id']: str(data['answer']) for data in data_list}
    with open(args.answers_file, 'r') as file:
        answers = [json.loads(line) for line in file]

    correct = []
    incorrect = []
    # temp = []
    for answer in answers:
        answer['GT'] = GTs[answer['question_id']]
        if GTs[answer['question_id']] == answer['text'].rstrip().lstrip():
            correct.append(answer)
        else:
            incorrect.append(answer)
        # temp.append((GTs[answer['question_id']], answer['text'].rstrip().lstrip()))

    print(f'Total: {len(correct)+len(incorrect)}, Correct: {len(correct)}, Accuracy: {len(correct) / (len(correct)+len(incorrect)) * 100:.2f}%')

    output_results = {}
    output_results['acc'] = len(correct) / len(correct)+len(incorrect) * 100
    output_results['correct'] = len(correct)
    output_results['count'] = len(correct)+len(incorrect)
    
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_result), exist_ok=True)

    with open(args.result_file, 'w') as f:
        json.dump(incorrect, f, indent=2)
        json.dump(correct, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(output_results, f, indent=2)
