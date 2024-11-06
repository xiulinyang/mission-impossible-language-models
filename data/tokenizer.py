from transformers import AutoTokenizer
from glob import glob
import json

data_path = 'babylm_multilingual'
file_path = f'{data_path}/EN.json'

all_text = []
lang_parse = json.load(open(file_path))

for sent in lang_parse:
    all_text.append(sent['sent_text'])

vocab = set(all_text)
old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
new_tokenizer = old_tokenizer.train_new_from_iterator(all_text, int(0.4*len(vocab)))
new_tokenizer.save_pretrained()
~                                
