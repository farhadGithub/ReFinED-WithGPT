import together
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")

dataset_type = 'dev'
dataset = f'wikiwebquestions_{dataset_type}_set'
margin = 2
n_epochs = 100
instruction = (f"Given a Wikidata query with resolved entities, generate the corresponding SPARQL. "
               f"Use property names instead of PIDs.")
model = {'model_output_name': f'msorichi@stanford.edu/llama-2-7b-margin-{margin}-epoch-{n_epochs}-2023-12-09-21-50-48'}
eval_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
             f'llm_training_wikipedia_model_wikidata_{dataset}_'
             f'processed_gpt_domain_not_passed_generic_examples_from_WWQ_compmix_margin_{margin}.json')
eval_set = json.load(open(eval_file))
output_file = (f'{os.environ.get("OUTPUT_FOLDER")}inference_{dataset}'
               f'_margin_{margin}_modified_{"_".join(model["model_output_name"].split("/"))}.json')

# check model exists
model_list = together.Models.list()
print(f"{len(model_list)} models available")
available_model_names = [model_dict['name'] for model_dict in model_list]
print(f"{model['model_output_name']} is ready: {model['model_output_name'] in available_model_names}")

#print(together.Finetune.retrieve(fine_tune_id=models[1]['fine_tune_id']))
#print(together.Finetune.get_job_status(fine_tune_id=models[1]['fine_tune_id']))

# deploy model
together.Models.start(model['model_output_name'])
print(f"{model['model_output_name']} is deployed: {together.Models.ready(model['model_output_name'])[0]['ready']}")
#test_prompt = "<s>[INST] <<SYS>> Given a Wikidata query with resolved entities, generate the corresponding SPARQL. Use property names instead of PIDs. <</SYS>> Query: what character did natalie portman play in star wars?\nEntities:Natalie Portman with QID Q37876;Star Wars with QID Q462 [/INST]</s>"
#test_prompt = f'''input: Given a Wikidata query with resolved entities, generate the corresponding SPARQL.
#                Use property names instead of PIDs. Query: what character did natalie portman
#                f"play in star wars?\nEntities:Natalie Portman with QID Q37876;Star Wars with QID Q462 output:'''

prompt = f'''input: Given a Wikidata query with resolved entities, generate the corresponding SPARQL.
               Use property names instead of PIDs.: Query: what kind of money to take to bahamas?
               \nEntities:The Bahamas with QID Q778'''
# inference
print(f'inference begins for eval file {os.path.basename(eval_file).split("/")[-1]} '
      f'and \nthe results will be written to {os.path.basename(output_file).split("/")[-1]}:')
output_dics = []
for i, item in enumerate(eval_set):
    question_id = item["id"]
    question_with_entities= item["input"]
    prompt = f'input: {instruction}: {question_with_entities} output: '
    output = together.Complete.create(
                prompt=prompt,
                model=model['model_output_name'],
                max_tokens=120,
                temperature=0.1,
                top_k=5,
                top_p=0.2,
                repetition_penalty=1.1,
                stop=['</s>']
    )
    print(f"{question_id}: {output['prompt'][0]}")
    response = output['output']['choices'][0]['text']
    if response.find('input') > 0:
        response = response[:response.find('input')]
    print(response)
    output_dic = {}
    output_dic['dev_set_id'] = question_id
    output_dic["predicted_sparql"] = response
    output_dics.append(output_dic)
    time.sleep(1)

with open(output_file, "w") as f:
    f.write(json.dumps(output_dics, indent=3))

#together.Models.stop(model['model_output_name'])

#model_list = together.Models.list()
#print(f"{len(model_list)} models available")
#available_model_names = [model_dict['name'] for model_dict in model_list]
#print(new_model_name in available_model_names)
