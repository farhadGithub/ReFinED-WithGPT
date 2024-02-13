from tqdm import tqdm
from dotenv import load_dotenv
import os
import json

dataset_type = 'dev'
dataset = f'wikiwebquestions_{dataset_type}_set'
margin = 0
load_dotenv()
size_of_predicted_entities_from_refined = True
if size_of_predicted_entities_from_refined:
    gpt_output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
                       f'llm_training_wikipedia_model_wikidata_{dataset}_'
                       f'processed_gpt_domain_not_passed_generic_examples_from_WWQ_compmix_margin_{margin}.json')
    output_file = f'{os.environ.get("DATASET_FOLDER")}llm_{dataset}_margin_{margin}_azure.json'
else:
    gpt_output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
                       f'llm_training_wikipedia_model_wikidata_{dataset}_'
                       f'processed_gpt_domain_not_passed_generic_examples_from_'
                       f'WWQ_compmix_max_entities_{margin}_extended_prompt.json')
    output_file = f'{os.environ.get("DATASET_FOLDER")}llm_{dataset}_max_entities_{margin}_azure.json'

gpt_outputs = json.load(open(gpt_output_file))

label_file = f'{os.environ.get("DATASET_FOLDER")}no_mention_refined.json'
labels = json.load(open(label_file))


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
        output_dic = {"instruction": instruction,
                      "input": question_with_entities,
                      "output": label,
                      "sparql": sparql,
                      "id": gpt_id}
        output_dics.append(output_dic)
else:
    for item in tqdm(gpt_outputs):
        question_id = item["id"]
        question_with_entities = item["input"]
        instruction = (f'Given a Wikidata query with resolved entities, generate the corresponding SPARQL.'
                       f'Use property names instead of PIDs.')
        output_dic = {"instruction": instruction,
                      "input": question_with_entities,
                      "id": question_id,
                      "output": "dummy"}
        output_dics.append(output_dic)

with open(output_file, "w") as f:
    f.write(json.dumps(output_dics, indent=3))
