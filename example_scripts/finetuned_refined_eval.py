from tqdm import tqdm
from dotenv import load_dotenv
import os
import csv

model_name = 'finetuned_refined'
target_domain = 'tvseries'

load_dotenv()
input_file = f'{os.environ.get("DATASET_FOLDER")}Compmix TV Questions Full Results  Small.csv'
dataset = 'compmix_dev_set'
total_exact_matches = 0.0
total_entity_count = 0.0
recall = []
precision = []

with open(input_file, 'r') as f:
    csvreader = csv.reader(f, delimiter=',')
    next(csvreader)
    for item in tqdm(csvreader):
        question = item[2]
        domain = item[1]
        if not (target_domain == 'all' or domain == target_domain):
            continue
        gold_entity_ids = []
        gold_entity_labels = []
        for i in [4, 6, 8]:
            if len(item[i]) > 0:
                gold_entity_labels.append(item[i])
                gold_entity_ids.append(item[i+1])
        total_entity_count += len(gold_entity_ids)
        row_id = item[0]
        num_matches = 0
        predicted_entities = []
        for i in [10, 12]:
            if len(item[i]) > 0:
                predicted_entity_label = item[i]
                predicted_entity_id = item[i+1]
                predicted_entities.append(predicted_entity_id)
                if predicted_entity_id in gold_entity_ids:
                    num_matches += 1
        total_exact_matches += num_matches
        recall.append(float(num_matches)/len(gold_entity_ids))
        if len(predicted_entities) > 0:
            precision.append(float(num_matches)/len(predicted_entities))
        else:
            precision.append(0.0)
f1 = [2*(p*r)/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]
macro_average_f1 = sum(f1)/len(f1)

print(f'For model {model_name} and dataset {dataset} and {target_domain}, exact match is {total_exact_matches} and'
      f' number of total entity is {total_entity_count} and EM ratio is {total_exact_matches/total_entity_count:0.2f}')
print(f'Macro average F1 score is {macro_average_f1:0.2f}')
