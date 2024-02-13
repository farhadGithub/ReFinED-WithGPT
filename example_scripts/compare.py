from tqdm import tqdm
from dotenv import load_dotenv
import os
import json

load_dotenv()
output_file = (f'{os.environ.get("OUTPUT_FOLDER")}'
               f'inference_epoch-15-max-entities-4-llama-2-with-alpaca.json')
data = json.load(open(output_file))

count = 0
for item in data:
    prediction = item['prediction']
    if len(prediction)>0:
        count += 1
print(count)


