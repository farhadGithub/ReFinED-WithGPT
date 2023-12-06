from refined.inference.processor import Refined
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
import csv
import argparse
from openai import OpenAI
from openai import AzureOpenAI


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
    metrics['micro_average_f1'] = ((2*metrics['micro_average_recall']*metrics['micro_average_precision']) /
                                   (metrics['micro_average_recall'] + metrics['micro_average_precision']))
    return metrics


def create_messages(prompt_type: str, domain: str, question: str) -> list:
    messages = []
    if prompt_type == 'gpt_domain_passed_domain_examples_from_compmix':
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
    elif prompt_type == 'gpt_domain_not_passed_domain_examples_from_compmix':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question and 
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
    elif prompt_type == 'gpt_domain_not_passed_generic_examples_from_compmix':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition system/entity disambiguation system. 
                                           You are given a question and 
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
    elif prompt_type == 'gpt_domain_not_passed_generic_examples_from_WWQ_compmix':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""You are a named entity recognition and entity disambiguation system. 
                                            You are given a question and you need to list all entities in the question 
                                            with a brief description for each entity. Each description should be max 10 
                                            words. Here are some examples:

            Question: what year lebron james came to the nba?
            Answer:
            
            1. LeBron James is American basketball player (born 1984)
            2. National Basketball Association is North American professional sports league 
            
            Question: what form of government was practiced in sparta?
            
            Answer:
            1. Sparta is city-state in ancient Greece
            
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

            1. Russell Crowe is New Zealand-born actor (born 1964)

            Question: what character did natalie portman play in star wars?
            
            1. natalie portman is Israeli-American actress and filmmaker
            2. star wars is epic space opera multimedia franchise created by George Lucas
            
            Question: what country is the grand bahama island in?
            
            1. Grand Bahama is island of the Bahamas
            
            Question: where are the nfl redskins from?
            
            1. Washington Commanders or Washington Redskins is American football team in the National Football League
            
            Question: what time zone am i in cleveland ohio?
            
            1. Cleveland is city in and county seat of Cuyahoga County, Ohio, United States
            
            Question: who is the prime minister of ethiopia?
            
            1. Ethiopia is country in the Horn of Africa"""},
            {"role": "user",
             "content": f"List the entities and their descriptions for this question:\n Question: {question}\n Answer:"}
        ]
    elif prompt_type == 'gpt_domain_passed_generic_examples_from_compmix':
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

#parser = argparse.ArgumentParser()
#parser.add_argument('--model_name', default='wikipedia_model',
#                    choices=['wikipedia_model', 'wikipedia_model_with_numbers'],
#                    help='Refined model name')
#parser.add_argument('--entity_set', default='wikidata', choices=['wikidata', 'wikipedia'],
#                    help='Refined entity set')
#parser.add_argument('--dataset_type', default='compmix', choices=['compmix', 'wikiwebquestions'],
#                    help='compmix or wikiwebquestions')
#parser.add_argument('--dataset_name', default='compmix_dev_set', help='dataset to evaluate')
#parser.add_argument('--target_domain', type=list, nargs='+',
#                    default="all", help='domains to evaluate')
#parser.add_argument('--prompt_type', default='gpt_domain_not_passed_generic_examples_from_compmix',
#                    help='Which prompt to choose for GPT')
#parser.add_argument('--save', default='False', type=bool,
#                   choices=['True', 'False'], help='Save results')
#args = parser.parse_args()
#model_name = args['model_name']

model_name = 'wikipedia_model'
entity_set = 'wikidata'
dataset_type = 'compmix'
dataset_name = 'compmix_dev_set'
#dataset_type = 'wikiwebquestions'
#dataset_name = 'wikiwebquestions_train_set_processed'
target_domains = ['all']
#target_domains = ['all', 'tvseries']
prompt_type = 'gpt_domain_not_passed_generic_examples_from_WWQ_compmix'
#prompt_type = 'gpt_domain_not_passed_generic_examples_from_compmix'
margin = 1  # additional number of entities compared to Refined original number that GPT can generate
instruction_for_query_generation = (f"Given a Wikidata query with resolved entities, "
                                    f"generate the corresponding SPARQL. "
                                    f"Use property names instead of PIDs.")
save = False
azure = True
seed = 12345
temperature = 0.0

print(f'Evaluating model {model_name} with entity set {entity_set} and dataset '
      f'{dataset_name} and {"and".join(target_domains)} domain(s) and with the prompt {prompt_type} and '
      f'{margin} additional predicted entities relative to what Refined generates:')

refined = Refined.from_pretrained(model_name=model_name, entity_set=entity_set)

load_dotenv()
if azure:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint="https://ovalopenairesource.openai.azure.com/")
else: #openai
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key)


input_file = f'{os.environ.get("DATASET_FOLDER")}{dataset_name}.json'
data = json.load(open(input_file))

result_file = \
    f'{os.environ.get("OUTPUT_FOLDER")}result_{model_name}_{entity_set}_{dataset_name}_{prompt_type}_margin_{margin}.csv'

llama_training_file = \
    f'{os.environ.get("OUTPUT_FOLDER")}llama_training_{model_name}_{entity_set}_{dataset_name}_{prompt_type}_margin_{margin}.csv'

if save:
    with open(result_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['question_id', 'domain', 'question',
                            'gold_entity_id', 'gold_entity_label',
                            'predicted_entity_id', 'predicted_entity_label', 'predicted_entity_score'])
    if dataset_type == 'wikiwebquestions':
        with open(llama_training_file, "w") as f:
            f.write(json.dumps({}, indent=3))

domains = []
num_exact_matches = []
num_predicted_entities = []
num_gold_entities = []
output_result = []
output_training_dics = []
num_no_span = 0
num_no_entity_in_span = 0
num_questions = 0

#for item in tqdm(data):
for item in data:
    num_questions += 1
    question = item['question']
    print(f'{num_questions}: {question}')
    if dataset_type == 'compmix':
        domain = item['domain']
        gold_entities = item['entities']
        gold_entity_ids = [entity['id'] for entity in gold_entities]
        gold_entity_labels = [entity['label'] for entity in gold_entities]
        question_id = item['question_id']
    else: # wikiwebquestions
        domain = 'all'
        gold_entities = None
        gold_entity_ids = item['entities']
        gold_entity_labels = [None for _ in range(len(gold_entity_ids))]
        question_id = item['question_id']
        sparql_query = item['sparql']

    num_gold_entities.append(len(gold_entity_ids))
    domains.append(domain)
    original_spans = refined.process_text(question)
    if azure:
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            seed=seed,
            temperature=temperature,
            messages=create_messages(prompt_type, domain, question)
        )
    else:  #openai
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
    num_exact_matches.append(0)
    num_predicted_entities.append(0)
    predicted_entity_ids = []
    predicted_entity_labels = []
    for gpt_description in gpt_descriptions[:min(len(gpt_descriptions), (len(original_spans) + margin))]:
        gpt_spans = refined.process_text(gpt_description)
        predicted_an_entity = False
        for gpt_span in gpt_spans:
            if gpt_span.predicted_entity is not None:
                predicted_entity = gpt_span.predicted_entity
                predicted_entity_id = predicted_entity.wikidata_entity_id
                predicted_entity_label = predicted_entity.wikipedia_entity_title
                if (predicted_entity_id in predicted_entity_ids) or (predicted_entity_id is None):
                    continue
                num_predicted_entities[-1] += 1
                predicted_an_entity = True
                predicted_entity_ids.append(predicted_entity_id)
                predicted_entity_labels.append(predicted_entity_label)
                if len(gpt_span.candidate_entities) > 0:
                    predicted_entity_score = gpt_span.candidate_entities[0][1]
                else:
                    predicted_entity_score = None
                if predicted_entity_id in gold_entity_ids:
                    num_exact_matches[-1] += 1
                    row = [question_id, domain, question,
                           predicted_entity_id, predicted_entity_label,
                           predicted_entity_id, predicted_entity_label, predicted_entity_score]
                    output_result.append(row)
                else:
                    row = [question_id, domain, question,
                           None, None,
                           predicted_entity_id, predicted_entity_label, predicted_entity_score]
                    output_result.append(row)
            if predicted_an_entity:
                break
    # print those gold entities that were not detected
    for i, gold_entity_id in enumerate(gold_entity_ids):
        if gold_entity_id not in predicted_entity_ids:
            row = [question_id, domain, question,
                   gold_entity_ids[i], gold_entity_labels[i], None, None, None]
            output_result.append(row)

    # generate output for training the LLM (Llama) for SPARQL query generation
    # works only with wikiwebquestions
    if dataset_type == 'wikiwebquestions':
        training_dic = {}
        entity_list = ';'.join([f'{label} with QID {qid}' for label,qid
                                in zip(predicted_entity_labels, predicted_entity_ids)])
        training_dic["input"] = f'Query: {question}\nEntities:{entity_list}'
        training_dic["sparql"] = sparql_query
        training_dic["id"] = question_id
        output_training_dics.append(training_dic)

    if save and num_questions % 100 == 0:
        with open(result_file, 'a') as f:
            csvwriter = csv.writer(f, delimiter=',')
            for row in output_result:
                csvwriter.writerow(row)
            output_result =[]
        if dataset_type == 'wikiwebquestions':
            with open(llama_training_file, "a") as f:
                f.write(json.dumps(output_training_dics, indent=3))
                output_training_dics = []

if save:
    with open(result_file, 'a') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for row in output_result:
            csvwriter.writerow(row)
    if dataset_type == 'wikiwebquestions':
        with open(llama_training_file, "a") as f:
            f.write(json.dumps(output_training_dics, indent=3))

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
          f'and {prompt_type} and margin {margin}:')
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
