import json
import tqdm
import argparse
import glob
import numpy as np
import ast
from llmlogscore import OpenAIClient

_SUFFIXES_TO_SCORE = [' yes', ' yeah']
_COMPLEMENT_SUFFIXES = [' no']

parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
parser.add_argument('--response_root', type=str, help='Response Root path.')
parser.add_argument('--openai_api_key', type=str, help='OpenAI API key.')
args = parser.parse_args()

_PROMPT = ('You are given a question, ground-truth answer, and a candidate answer. '
           'Question: <question> \nGround-truth answer: <GT> \nCandidate answer: <answer> \n'
           'Is the semantic meaning of the ground-truth and candidate answers similar? Answer in one word - Yes or No.')

def calculate_all_metrics(response_root, client, prompt):
    score = 0
    total = 0
    failed_parsing = 0

    for file in tqdm.tqdm(glob.glob(response_root + '/*.json')):
        with open(file, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            question = value.get('question', '')
            ground_truth = value.get('answer', '')
            raw_response = value.get('response', '')
            try:
                # remove markdown code block markers
                response_str = raw_response.strip()
                if response_str.startswith("```json"):
                    response_str = response_str[len("```json"):].strip()
                if response_str.endswith("```"):
                    response_str = response_str[:-len("```")].strip()
                response_dict = ast.literal_eval(response_str)
                candidate_answer = response_dict.get('Answer', '')
                
                total += 1
                current_prompt = prompt.replace('<question>', question) \
                                       .replace('<GT>', ground_truth) \
                                       .replace('<answer>', candidate_answer)
                response, prob_yes = client.call_openai_with_score(
                    prompt=current_prompt,
                    suffixes=_SUFFIXES_TO_SCORE,
                    complement_suffixes=_COMPLEMENT_SUFFIXES,
                    output_prefix=''
                )
                score += prob_yes

            except Exception:
                failed_parsing += 1
                total += 1

    print('Printing Metric ..')
    print('Metric: ', score / total)
    print("Examples with Failed Parsing: {}".format(failed_parsing))
    print("all: ", total)

with open("~/openai_api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAIClient(
    model_name='gpt-4o',
    api_key=api_key,
    json_output_path='./saved_output_l3score/',
)

calculate_all_metrics('../responses/gemini_response/', client, _PROMPT)