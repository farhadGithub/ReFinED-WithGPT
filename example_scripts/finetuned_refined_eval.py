from tqdm import tqdm
from dotenv import load_dotenv
import os
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


model_name = 'finetuned_refined'
target_domains = ['tvseries']

load_dotenv()
input_file = f'{os.environ.get("DATASET_FOLDER")}Compmix TV Questions Full Results  Small.csv'
dataset = 'compmix_dev_set'

domains = []
num_exact_matches = []
num_predicted_entities = []
num_gold_entities = []

with open(input_file, 'r') as f:
    csvreader = csv.reader(f, delimiter=',')
    next(csvreader)
    for item in tqdm(csvreader):
        question = item[2]
        domain = item[1]
        domains.append(domain)
        gold_entity_ids = []
        gold_entity_labels = []
        for i in [4, 6, 8]:
            if len(item[i]) > 0:
                gold_entity_labels.append(item[i])
                gold_entity_ids.append(item[i+1])
        row_id = item[0]
        num_exact_matches.append(0)
        predicted_entities = []
        for i in [10, 12]:
            if len(item[i]) > 0:
                predicted_entity_label = item[i]
                predicted_entity_id = item[i+1]
                predicted_entities.append(predicted_entity_id)
                if predicted_entity_id in gold_entity_ids:
                    num_exact_matches[-1] += 1

        num_gold_entities.append(len(gold_entity_ids))
        num_predicted_entities.append(len(predicted_entities))


for target_domain in target_domains:
    metrics = compute_metrics(num_exact_matches=num_exact_matches,
                              num_predicted_entities=num_predicted_entities,
                              num_gold_entities=num_gold_entities,
                              domains=domains,
                              target_domain=target_domain)
    print(f'==={target_domain}===')
    print(f'For model {model_name} and dataset {dataset} and {target_domain} domain(s):')
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
