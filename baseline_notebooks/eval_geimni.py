from google import genai
import random
import json
import os
import glob
import time
from dotenv import load_dotenv
import os

# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="google/spiqa", filename="test-A/SPIQA_testA.json", repo_type="dataset", local_dir='./data')
# hf_hub_download(repo_id="google/spiqa", filename="test-A/SPIQA_testA_Images.zip", repo_type="dataset", local_dir='./data')


load_dotenv()

REQUEST_LIMIT = 15
REQUEST_INTERVAL = 60  
# QUESTION_TYPE = "image"
EVAL_DATA_PATH = '../IDL-Project/data/test-A/SPIQA_testA.json'
EVAL_IMAGE_PATH = '../IDL-Project/data/test-A/SPIQA_testA_Images'
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RESPONSE_PATH = "response"
IMAGE_RES = 224
PROMPT = "You are given a question, and a few input captions. \
      Please answer the question based on the input captions. \
        Question: <question>. Output in the following format: {'Answer': 'Direct Answer to the Question'}. \n"
# if QUESTION_TYPE == "image":
#    PROMPT = "You are given a question, and a few input images. \
#     Please answer the question based on the input images. \
#       Question: <question>. Output in the following format: {'Answer': 'Direct Answer to the Question'}. \n"
# elif QUESTION_TYPE == "caption":
    # PROMPT = "You are given a question, and a few input captions. \
    #   Please answer the question based on the input captions. \
    #     Question: <question>. Output in the following format: {'Answer': 'Direct Answer to the Question'}. \n"

client = genai.Client(api_key=GEMINI_API_KEY)

with open(EVAL_DATA_PATH, "r") as f:
  eval_data = json.load(f)

def prepare_inputs_only_caption(paper, question_idx):
  all_figures = list(paper['all_figures'].keys())
  referred_figures = [paper['qa'][question_idx]['reference']]
  answer = paper['qa'][question_idx]['answer']
  all_figures_captions = []

  if len(all_figures) > 8:
    referred_figures_number = len(referred_figures)
    other_figures_number = 8 - referred_figures_number
    all_other_figures = list(set(all_figures) - set(referred_figures))
    random.shuffle(all_other_figures)
    all_figures_modified = all_other_figures[:other_figures_number] + referred_figures
    random.shuffle(all_figures_modified)
    referred_figures_indices = [all_figures_modified.index(element) for element in referred_figures]
  else:
    all_figures_modified = all_figures
    random.shuffle(all_figures_modified)
    referred_figures_indices = [all_figures_modified.index(element) for element in referred_figures]

  for figure in all_figures_modified:
    all_figures_captions.append(paper['all_figures'][figure]['caption'])

  return answer, all_figures_captions, referred_figures_indices, all_figures_modified, referred_figures


def infer_gemini(data):
  
    os.makedirs(RESPONSE_PATH, exist_ok=True)

    request_count = 0
    start_time = time.time()

    for paper_id, paper in data.items():
        if os.path.exists(os.path.join(RESPONSE_PATH, str(paper_id) + '_response.json')):
            continue
        response_paper = {}

        try:
            for question_idx, qa in enumerate(paper['qa']):
                
                if request_count >= REQUEST_LIMIT:
                  time_left = time.time() - start_time
                  if time_left < REQUEST_INTERVAL:
                      time.sleep(REQUEST_INTERVAL - time_left) 
                  request_count = 0
                  start_time = time.time()
                
                question = qa['question']

                answer, all_figures_captions, referred_figures_indices, all_figures_modified, referred_figures = prepare_inputs_only_caption(paper, question_idx)

                contents = [PROMPT.replace('<question>', question)]

                for idx, _ in enumerate(all_figures_captions):
                    contents.append("Caption {}: {}".format(idx, all_figures_captions[idx]))
                    contents.append('\n\n')

                response = client.models.generate_content(model = GEMINI_MODEL,contents=contents)
                print(response.text)
                print('---------------------------------------------------------------------------------------------------------')

                time.sleep(4)

                response_paper.update({question_idx: {'question': question, 'referred_figures_indices':             
                                                      referred_figures_indices, 'response': response.text,
                                                      'all_figures_names': all_figures_modified, 
                                                      'referred_figures_names': referred_figures, 'answer': answer}})
                request_count += 1

        except Exception as e:
            print('Error in Generating ...')
            print(e)
            continue

        with open(os.path.join(RESPONSE_PATH, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)


if __name__ == '__main__':
    infer_gemini(eval_data)
    print(len(glob.glob(RESPONSE_PATH + '/*.json')))