from refined.inference.processor import Refined
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import csv


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


model_name = 'wikipedia_model'
entity_set = 'wikidata'
#dataset_type = 'wikiwebquestions'
#dataset_name = 'wikiwebquestions_train_set_processed'
dataset_type = 'compmix'
dataset_name = 'compmix_dev_set'
target_domains = ['all']
#target_domains = ['all', 'tvseries']
other_flavors = 'no_domain_added_to_questions'
save = True

print(f'Evaluating model {model_name} with entity set {entity_set} and dataset '
      f'{dataset_name} and {" and ".join(target_domains)} domain(s) and with {other_flavors}:')

refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)

load_dotenv()
input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset_name}.json'
data = json.load(open(input_file))

output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
               f'{model_name}_{entity_set}_{dataset_name}_{other_flavors}.csv')

domains = []
num_exact_matches = []
num_predicted_entities = []
num_gold_entities = []
num_no_span = 0
num_no_entity_in_span = 0
num_questions = 0
output = []
#accuracy_data = dict()

for item in tqdm(data):
    num_questions += 1
    if dataset_type == 'compmix':
        question = item['question']
        domain = item['domain']
        gold_entities = item['entities']
        gold_entity_ids = [entity['id'] for entity in gold_entities]
        gold_entity_labels = [entity['label'] for entity in gold_entities]
        question_id = item['question_id']
    else: # wikiwebquestions
        question = item['question']
        domain = 'all'
        gold_entities = None
        gold_entity_ids = item['entities']
        gold_entity_labels = [None for _ in range(len(gold_entity_ids))]
        question_id = item['question_id']
    domains.append(domain)

    if other_flavors == 'domain_added_to_questions':
        spans = refined.process_text(domain + ': ' + question)
    else:
        spans = refined.process_text(question)

    if len(spans) == 0:
        num_no_span += 1

    num_exact_matches.append(0)
    num_predicted_entities.append(0)
    predicted_entity_ids = set()
    for span in spans:
        predicted_entity = span.predicted_entity
        if predicted_entity is not None:
            predicted_entity_id = predicted_entity.wikidata_entity_id
            predicted_entity_label = predicted_entity.wikipedia_entity_title
            if (predicted_entity_id in predicted_entity_ids) or (predicted_entity_id is None):
                continue
            num_predicted_entities[-1] += 1
            predicted_entity_ids.add(predicted_entity_id)
            if len(span.candidate_entities) > 0:
                predicted_entity_score = span.candidate_entities[0][1]
            else:
                predicted_entity_score = None
            if predicted_entity_id in gold_entity_ids:
                num_exact_matches[-1] += 1
                row = [question_id, domain, question,
                       predicted_entity_id, predicted_entity_label,
                       predicted_entity_id, predicted_entity_label, predicted_entity_score]
                output.append(row)
            else:
                row = [question_id, domain, question,
                       None, None,
                       predicted_entity_id, predicted_entity_label, predicted_entity_score]
                output.append(row)
        else:
            num_no_entity_in_span += 1

    # print those gold entities that were not detected
    for i, gold_entity_id in enumerate(gold_entity_ids):
        if gold_entity_id not in predicted_entity_ids:
            row = [question_id, domain, question,
                    gold_entity_ids[i], gold_entity_labels[i], None, None, None]
            output.append(row)

    num_gold_entities.append(len(gold_entity_ids))

    #print(f'{question_id}: {question}')
    #print(f'    Number of gold entities: {len(gold_entity_ids)}')
    #print(f'    Number of predicted entities: {len(predicted_entity_ids)}')
    #print(f'    Number of exact matches: {num_exact_matches[-1]}')
    #if len(gold_entity_ids) not in accuracy_data:
    #    accuracy_data[len(gold_entity_ids)] = dict()
    #    accuracy_data[len(gold_entity_ids)]['num of gold entities'] = 0
    #    accuracy_data[len(gold_entity_ids)]['num of predicted entities'] = 0
    #    accuracy_data[len(gold_entity_ids)]['num of exact matches'] = 0
    #accuracy_data[len(gold_entity_ids)]['num of gold entities'] += len(gold_entity_ids)
    #accuracy_data[len(gold_entity_ids)]['num of predicted entities'] += len(predicted_entity_ids)
    #accuracy_data[len(gold_entity_ids)]['num of exact matches'] += num_exact_matches[-1]

if save:
    with open(output_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['question_id', 'domain', 'question',
                            'gold_entity_id', 'gold_entity_label',
                            'predicted_entity_id', 'predicted_entity_label', 'predicted_entity_score'])
        for row in output:
            csvwriter.writerow(row)

#for k in accuracy_data:
#    print(f'When number of gold entities per question is {k}:')
#    print(f"    Total number of gold entities is: {accuracy_data[k]['num of gold entities']}")
#    print(f"    Total number of predicted entities is: {accuracy_data[k]['num of predicted entities']}")
#    print(f"    Total number of exact matches is: {accuracy_data[k]['num of exact matches']}")

print(f'Number of questions with no span: {num_no_span}')
print(f'Number of spans with no entity: {num_no_entity_in_span}')

for target_domain in target_domains:
    metrics = compute_metrics(num_exact_matches=num_exact_matches,
                              num_predicted_entities=num_predicted_entities,
                              num_gold_entities=num_gold_entities,
                              domains=domains,
                              target_domain=target_domain)
    print(f'==={target_domain}===')
    print(f'For model {model_name} with entity set {entity_set} and dataset {dataset_name} and {target_domain} domain(s) '
          f'and {other_flavors}:')
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
