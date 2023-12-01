from refined.inference.processor import Refined
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import csv
from openai import OpenAI


def create_messages(prompt_type: str, target_domain: str, question: str) -> list:
    messages = []
    if prompt_type == 'gpt_domain_passed_examples_from_domain':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question from a domain (such as music, tv series) and 
                                           you need to list all entities in the question with a brief description of 
                                           each entity. Each description should be max 10 words. Here are some examples:

            Question from tv series domain: What is the genre of the tv series High Seas?
            Answer:

            1. High Seas is a Spanish television series

            Question from tv series domain: Which country did the TV series Coupling originate?
            Answer:

            1. Coupling is a British television series (2000–2004)"""},
            {"role": "user",
             "content": f"List the entities and their descriptions in this question:\n "
                        f"Question from {target_domain} domain: {question}\n Answer:"}
        ]
    elif prompt_type == 'gpt_domain_not_passed_examples_from_domain':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question from a domain (such as music, tv series) and 
                                           you need to list all entities in the question with a brief description of 
                                           each entity. Each description should be max 10 words. Here are some examples:

            Question: What is the genre of the tv series High Seas?
            Answer:

            1. High Seas is a Spanish television series

            Question: Which country did the TV series Coupling originate?
            Answer:

            1. Coupling is a British television series (2000–2004)"""},
            {"role": "user",
             "content": f"List the entities and their descriptions for this question:\n Question: {question}\n Answer:"}
        ]
    elif prompt_type == 'gpt_domain_not_passed_examples_generic':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question from a domain (such as music, tv series) and 
                                           you need to list all entities in the question with a brief description of 
                                           each entity. Each description should be max 10 words. Here are some examples:

            Question: What is the genre of the tv series High Seas?
            Answer:

            1. High Seas is a Spanish television series

            Question: Which country did the TV series Coupling originate?
            Answer:

            1. Coupling is a British television series (2000–2004)
             
            Question: What year was M.O.V.E first formed?
            Answer:

            1. M.O.V.E is a Japanese musical group

            Question: What year was the inception of the soccer club Manchester United F.C.?
            
            1. Manchester United F.C. is association football club in Manchester, England
            
            Question: What is Russell Crowe's date of birth?
            
            1. Russell Crowe is New Zealand-born actor (born 1964)"""},
            {"role": "user",
             "content": f"List the entities and their descriptions for this question:\n Question: {question}\n Answer:"}
        ]
    elif prompt_type == 'gpt_domain_passed_examples_generic':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question from a domain (such as music, tv series) and 
                                           you need to list all entities in the question with a brief description of 
                                           each entity. Each description should be max 10 words. Here are some examples:

            Question from tv series domain: What is the genre of the tv series High Seas?
            Answer:

            1. High Seas is a Spanish television series

            Question from tv series domain: Which country did the TV series Coupling originate?
            Answer:

            1. Coupling is a British television series (2000–2004)
             
            Question from music domain: What year was M.O.V.E first formed?
            Answer:

            1. M.O.V.E is a Japanese musical group

            Question from soccer domain: What year was the inception of the soccer club Manchester United F.C.?
            
            1. Manchester United F.C. is association football club in Manchester, England
            
            Question from movies domain: What is Russell Crowe's date of birth?
            
            1. Russell Crowe is New Zealand-born actor (born 1964)"""},
            {"role": "user",
             "content": f"List the entities and their descriptions in this question:\n "
                        f"Question from {target_domain} domain: {question}\n Answer:"}
        ]
    return messages


model_name = 'wikipedia_model'
entity_set = 'wikidata'
dataset = 'compmix_dev_set'
target_domain = 'all'
prompt_type = 'gpt_domain_passed_examples_generic'
seed = 12345
temperature = 0.0
save = False

refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset}.json'
data = json.load(open(input_file))

output_file = f'{os.environ.get("OUTPUT_FOLDER")}{model_name}_{entity_set}_{dataset}_{target_domain}_{prompt_type}.csv'

total_exact_matches = 0.0
total_entity_count = 0.0
num_questions = 0
recall = []
precision = []
output = []

for item in tqdm(data):
    question = item['question']
    domain = item['domain']
    if not (target_domain == 'all' or domain == target_domain):
        continue
    gold_entities = item['entities']
    gold_entity_ids = [entity['id'] for entity in gold_entities]
    gold_entity_labels = [entity['label'] for entity in gold_entities]
    total_entity_count += len(gold_entity_ids)
    question_id = item['question_id']
    convmix_question_id = item['convmix_question_id']
    original_spans = refined.process_text(question)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=seed,
        temperature=temperature,
        messages=create_messages(prompt_type, target_domain, question)
    )

    gpt_response = response.choices[0].message.content
    gpt_descriptions = gpt_response.split("\n")
    # drop item number
    gpt_descriptions = [gpt_description[3:] for gpt_description in gpt_descriptions if len(gpt_description) > 3]
    if len(original_spans) > 0:
        gpt_descriptions = gpt_descriptions[:len(original_spans)]
    gpt_entities = 0
    num_matches = 0
    for gpt_description in gpt_descriptions:
        gpt_spans = refined.process_text(gpt_description)
        for gpt_span in gpt_spans:
            if gpt_span.predicted_entity is not None:
                if gpt_entities > len(original_spans):
                    continue
                predicted_entity = gpt_span.predicted_entity
                predicted_entity_id = predicted_entity.wikidata_entity_id
                predicted_entity_label = predicted_entity.wikipedia_entity_title
                if len(gpt_span.candidate_entities) > 0:
                    predicted_entity_score = gpt_span.candidate_entities[0][1]
                else:
                    predicted_entity_score = None
                if predicted_entity_id in gold_entity_ids:
                    num_matches += 1
                    row = [question_id, convmix_question_id, question,
                           predicted_entity_id, predicted_entity_label,
                           predicted_entity_id, predicted_entity_label, predicted_entity_score]
                    output.append(row)
                if predicted_entity_id is not None:  # only consider one entity per
                    gpt_entities += 1
                    break
    if num_matches == 0:
        for i in range(len(gold_entity_labels)):
            row = [question_id, convmix_question_id, question,
                   gold_entity_ids[i], gold_entity_labels[i], None, None, None]
            output.append(row)
    total_exact_matches += num_matches
    recall.append(float(num_matches)/len(gold_entity_ids))
    if gpt_entities > 0:
        precision.append(float(num_matches)/gpt_entities)
    else:
        precision.append(0.0)

if save:
    with open(output_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['question_id', 'convmix_question_id', 'question',
                            'gold_entity_id', 'gold_entity_label',
                            'predicted_entity_id', 'predicted_entity_label', 'predicted_entity_score'])
        for row in output:
            csvwriter.writerow(row)

f1 = [2*(p*r)/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]
macro_average_f1 = sum(f1)/len(f1)
print(f'Number of questions: {num_questions}')
print(f'For model {model_name} and entity set {entity_set} and dataset {dataset} and {target_domain} and {prompt_type},'
      f'exact match is {total_exact_matches} and'
      f' number of total entity is {total_entity_count} and EM ratio is {total_exact_matches/total_entity_count:.2f}')
print(f'Macro average F1 score is {macro_average_f1:0.2f}')
