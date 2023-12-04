from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import re


dataset = 'wikiwebquestions_dev_set'
load_dotenv()
input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset}.json'
data = json.load(open(input_file))
output_file = f'{os.environ.get("DATASET_FOLDER")}{dataset}_processed.json'

num_questions = 0
dictionaries = []
for item in data:
    num_questions += 1
    sparql = item['sparql']
    entities = re.findall(r'Q\d+', sparql)
    dictionary = {
        "question_id": item['id'],
        "question": item['utterance'],
        "entities": entities
    }
    dictionaries.append(dictionary)

with open(output_file, "w") as f:
    f.write(json.dumps(dictionaries, indent=3))
