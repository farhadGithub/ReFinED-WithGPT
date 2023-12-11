import together
import os
from dotenv import load_dotenv

margin = 0
base_model_name = 'togethercomputer/llama-2-7b'
#base_model_name = 'togethercomputer/llama-2-7b-chat'
models = [{'file_id': 'file-4909ed72-3a33-4658-ab75-4314efd6bab8',
           'file_name': 'llm_wikiwebquestions_train_set_margin_1_modified.jsonl'}]

load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")
n_epochs = 100
batch_size=20
ft_resp = together.Finetune.create(
  training_file=models[margin]['file_id'],
  model=base_model_name,
  n_epochs=n_epochs,
  batch_size=batch_size,
  n_checkpoints=4,
  learning_rate=2e-5,
  suffix=f'margin-0-epoch-{n_epochs}-{batch_size}',
)

fine_tune_id = ft_resp['id']
print(fine_tune_id)
print(ft_resp)
