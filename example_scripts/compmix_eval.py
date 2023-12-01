from refined.inference.processor import Refined
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import csv

model_name = 'wikipedia_model'
entity_set = 'wikidata'
dataset = 'compmix_dev_set'
target_domain = 'all'
other_flavors = 'domain_added_to_questions'
save = False

refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)

load_dotenv()
input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset}.json'
data = json.load(open(input_file))

output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
               f'{model_name}_{entity_set}_{dataset}_{target_domain}_{other_flavors}.csv')

total_exact_matches = 0.0
total_entity_count = 0.0
num_no_span = 0
num_no_entity_in_span = 0
num_questions = 0
recall = []
precision = []
output = []
for item in tqdm(data):
    num_questions += 1
    question = item['question']
    domain = item['domain']
    if not (target_domain == 'all' or domain == target_domain):
        continue
    gold_entities = item['entities']
    gold_entity_ids = [entity['id'] for entity in gold_entities]
    gold_entity_labels = [entity['label'] for entity in gold_entities]
    total_entity_count += len(gold_entity_ids)
    question_id = item['question_id']
    compmix_question_id = item['convmix_question_id']
    if other_flavors == 'domain_added_to_questions':
        spans = refined.process_text(domain + ': ' + question)
    else:
        spans = refined.process_text(question)
    if len(spans) == 0:
        num_no_span += 1
    num_matches = 0
    predicted_entities = []
    for span in spans:
        predicted_entity = span.predicted_entity
        if predicted_entity is not None:
            predicted_entity_id = predicted_entity.wikidata_entity_id
            predicted_entity_label = predicted_entity.wikipedia_entity_title
            if predicted_entity_id is not None:
                predicted_entity_score = span.candidate_entities[0][1]
                predicted_entities.append(predicted_entity_id)
                if predicted_entity_id in gold_entity_ids:
                    num_matches += 1
                    row = [question_id, compmix_question_id, question,
                           predicted_entity_id, predicted_entity_label,
                           predicted_entity_id, predicted_entity_label, predicted_entity_score]
        else:
            num_no_entity_in_span += 1
    if num_matches == 0:
        for i in range(len(gold_entity_labels)):
            row = [question_id, compmix_question_id, question,
                   gold_entity_ids[i], gold_entity_labels[i], None, None, None]
    total_exact_matches += num_matches
    recall.append(float(num_matches)/len(gold_entity_ids))
    if len(predicted_entities) > 0:
        precision.append(float(num_matches)/len(predicted_entities))
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
print(f'Number of questions with no span: {num_no_span}')
print(f'Number of spans with no entity: {num_no_entity_in_span}')
print(f'For model {model_name} with entity set {entity_set} and dataset {dataset} and {target_domain} domain(s) and '
      f'{other_flavors}, exact match is {total_exact_matches} and'
      f' number of total entity is {total_entity_count} and EM ratio is {total_exact_matches/total_entity_count:0.2f}')
print(f'Macro average F1 score is {macro_average_f1:0.2f}')
