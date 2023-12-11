from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import pandas as pd

#B_INST, E_INST = "[INST]", "[/INST]"
#B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#DEFAULT_SYSTEM_PROMPT = """\
#You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

#def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
#    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
#   prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
#    return prompt_template


dataset_type = 'train'
dataset = f'wikiwebquestions_{dataset_type}_set'
margin = 2
load_dotenv()
gpt_output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
                   f'llm_training_wikipedia_model_wikidata_{dataset}_'
                   f'processed_gpt_domain_not_passed_generic_examples_from_WWQ_compmix_margin_{margin}.json')

gpt_outputs = json.load(open(gpt_output_file))

label_file = f'{os.environ.get("DATASET_FOLDER")}no_mention_refined.json'
labels = json.load(open(label_file))
output_file = f'{os.environ.get("DATASET_FOLDER")}llm_{dataset}_margin_{margin}_modified.jsonl'

output_dics = []
if dataset_type == 'train':
    i = 0
    for item in tqdm(labels):
        question_id = item["id"]
        label = item["output"]
        instruction = item["instruction"]
        sparql = item["sparql"]
        while i < len(gpt_outputs):
            gpt_id = gpt_outputs[i]["id"]
            question_with_entities = gpt_outputs[i]["input"]
            i += 1
            if gpt_id == question_id:
                break
        #prompt = get_prompt(instruction,)
        #output_dic = {"text": f'<s>[INST] <<SYS>> {instruction} <</SYS>> {question_with_entities} [/INST]{label}</s>'}
        output_dic = {"text": f'input: {instruction}: {question_with_entities} output: {label}'}
        output_dics.append(output_dic)
else:
    for item in tqdm(gpt_outputs):
        question_id = item["id"]
        question_with_entities = item["input"]
        instruction = (f'Given a Wikidata query with resolved entities, generate the corresponding SPARQL.'
                       f'Use property names instead of PIDs.')
        output_dic = {"text": f'input: {instruction}: {question_with_entities} output: '}
        output_dics.append(output_dic)

with open(output_file, "w") as f:
    l = len(output_dics) - 1
    for i, output_dic in enumerate(output_dics):
        if i != l:
            f.write(json.dumps(output_dic)+"\n")
        else:
            f.write(json.dumps(output_dic))
