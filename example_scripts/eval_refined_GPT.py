from refined.inference.processor import Refined
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import csv
from openai import OpenAI

def compute_metrics(num_exact_matches: list, num_predicted_entities: list,
                    num_gold_entities: list, domains: list,
                    target_domain: str = 'all') -> dict:
    metrics = dict()
    total_questions = 0
    total_exact_matches = 0
    total_gold_entities = 0
    total_predicted_entities = 0
    recall = []
    precision = []
    for e, p, g, d in zip(num_exact_matches, num_predicted_entities, num_gold_entities, domains):
        if d == target_domain or target_domain == 'all':
            total_questions += 1
            total_exact_matches += e
            total_predicted_entities += p
            total_gold_entities += g
            recall.append(float(e)/g)
            if p == 0:
                precision.append(0)
            else:
                precision.append(float(e)/p)
    f1 = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    metrics['total_questions'] = total_questions
    metrics['total_exact_matches'] = total_exact_matches
    metrics['total_gold_entities'] = total_gold_entities
    metrics['total_predicted_entities'] = total_predicted_entities
    metrics['macro_average_recall'] = sum(recall) / total_questions
    metrics['macro_average_precision'] = sum(precision) / total_questions
    metrics['macro_average_f1'] = sum(f1) / total_questions
    metrics['micro_average_recall'] = total_exact_matches / total_gold_entities
    metrics['micro_average_precision'] = total_exact_matches / total_predicted_entities
    metrics['micro_average_f1'] = ((2*metrics['micro_average_recall']*metrics['micro_average_precision'])/
                                   (metrics['micro_average_recall'] + metrics['micro_average_precision']))
    return metrics


def create_messages(prompt_type: str, domain: str, question: str) -> list:
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
                        f"Question from {domain} domain: {question}\n Answer:"}
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
                        f"Question from {domain} domain: {question}\n Answer:"}
        ]
    return messages


model_name = 'wikipedia_model'
entity_set = 'wikidata'
dataset = 'compmix_dev_set'
target_domains = ['all', 'tvseries']
prompt_type = 'gpt_domain_not_passed_examples_generic'
seed = 12345
temperature = 0.0
save = False

refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset}.json'
data = json.load(open(input_file))

output_file = f'{os.environ.get("OUTPUT_FOLDER")}{model_name}_{entity_set}_{dataset}_{prompt_type}.csv'

domains = []
num_exact_matches = []
num_predicted_entities = []
num_gold_entities = []
output = []
num_no_span = 0
num_no_entity_in_span = 0
num_questions = 0

#for item in tqdm(data):
for item in data:
    num_questions += 1
    question = item['question']
    print(f'{num_questions}: {question}')
    domain = item['domain']
    domains.append(domain)
    gold_entities = item['entities']
    gold_entity_ids = [entity['id'] for entity in gold_entities]
    gold_entity_labels = [entity['label'] for entity in gold_entities]
    num_gold_entities.append(len(gold_entity_ids))
    question_id = item['question_id']
    convmix_question_id = item['convmix_question_id']
    original_spans = refined.process_text(question)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=seed,
        temperature=temperature,
        messages=create_messages(prompt_type, domain, question)
    )

    gpt_response = response.choices[0].message.content
    gpt_descriptions = gpt_response.split("\n")
    # drop item number
    gpt_descriptions = [gpt_description[3:] for gpt_description in gpt_descriptions if len(gpt_description) > 3]
    if len(original_spans) > 0:
        gpt_descriptions = gpt_descriptions[:len(original_spans)]
    gpt_entities = 0
    num_exact_matches.append(0)
    num_predicted_entities.append(0)
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
                    num_exact_matches[-1] += 1
                    row = [question_id, convmix_question_id, domain, question,
                           predicted_entity_id, predicted_entity_label,
                           predicted_entity_id, predicted_entity_label, predicted_entity_score]
                    output.append(row)
                if predicted_entity_id is not None:  # only consider one entity per
                    num_predicted_entities[-1] += 1
                    break

    if num_exact_matches[-1] == 0:
        for i in range(len(gold_entity_labels)):
            row = [question_id, convmix_question_id, domain, question,
                   gold_entity_ids[i], gold_entity_labels[i], None, None, None]
            output.append(row)

if save:
    with open(output_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['question_id', 'convmix_question_id', 'domain', 'question',
                            'gold_entity_id', 'gold_entity_label',
                            'predicted_entity_id', 'predicted_entity_label', 'predicted_entity_score'])
        for row in output:
            csvwriter.writerow(row)

print(f'Number of questions with no span: {num_no_span}')
print(f'Number of spans with no entity: {num_no_entity_in_span}')

for target_domain in target_domains:
    metrics = compute_metrics(num_exact_matches=num_exact_matches,
                              num_predicted_entities=num_predicted_entities,
                              num_gold_entities=num_gold_entities,
                              domains=domains,
                              target_domain=target_domain)
    print(f'==={target_domain}===')
    print(f'For model {model_name} with entity set {entity_set} and dataset {dataset} and {target_domain} domain(s) '
          f'and {prompt_type}:')
    print(f'Number of questions: {metrics["total_questions"]}')
    print(f'Number of exact matches is {metrics["total_exact_matches"]}')
    print(f'Number of gold entities is {metrics["total_gold_entities"]}')
    print(f'Number of predicted entities is {metrics["total_predicted_entities"]}')
    print(f'EM/micro average recall is {metrics["micro_average_recall"]:0.2f}')
    print(f'Micro average precision is {metrics["micro_average_precision"]:0.2f}')
    print(f'Micro average F1 is {metrics["micro_average_f1"]:0.2f}')
    print(f'Macro average recall is {metrics["macro_average_recall"]:0.2f}')
    print(f'Macro average precision is {metrics["macro_average_precision"]:0.2f}')
    print(f'Macro average F1 is {metrics["macro_average_f1"]:0.2f}')
    print(f'')
