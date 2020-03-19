from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer
)
import re
import torch
# Change for sequence classification model
from model_bert_similar_dist import BertModel

MODEL_CLASSES  = {
    "bert" : (
        BertConfig,
        BertModel,
        BertTokenizer,
    ),
    "distilbert":(
        DistilBertConfig,
        DistilBertModel,
        DistilBertTokenizer,
    )
}
# Initializate config classes
config_class_target, model_class_target, tokenizer_class_target = MODEL_CLASSES['bert']
config_class_source, model_class_source, tokenizer_class_source = MODEL_CLASSES['distilbert']
config_bert = config_class_target.from_pretrained('./output_bert')
config_dist = config_class_source.from_pretrained('./output_dist')
# Initialize models
model_dist = model_class_source.from_pretrained('distilbert-base-multilingual-cased')
model_bert = model_class_target(config_bert)
# Initialize tokenizers
tokenizer = tokenizer_class_source.from_pretrained('distilbert-base-multilingual-cased')
print('Number of layer parameters distillation model: ')
print(len(model_dist.state_dict().keys()))
print('Number of layer parameters Bert model: ')
print(len(model_bert.state_dict().keys()))
n_matches = 0
state_dict_target = model_bert.state_dict().copy()
use_prefix_bert = 'bert.embeddings.token_type_embeddings.weight' in model_bert.state_dict().keys()
if use_prefix_bert:
    prefix_bert = 'bert.'
else:
    prefix_bert = ''
use_prefix_distil = 'distilbert.embeddings.word_embeddings.weight' in model_dist.state_dict().keys()
if use_prefix_distil:
    prefix_distil = "distilbert\."
else:
    prefix_distil = ""
for l_name in model_dist.state_dict().keys():
    r1 = re.compile(r""+prefix_distil+"embeddings")
    r2 = re.compile(r""+prefix_distil+"transformer")
    r3 = re.compile(r"q_lin")
    r4 = re.compile(r"v_lin")
    r5 = re.compile(r"k_lin")
    r6 = re.compile(r"out_lin")
    r7 = re.compile(r"sa_layer_norm")
    r8 = re.compile(r"ffn.lin1")
    r9 = re.compile(r"ffn.lin2")
    r10 = re.compile(r"output_layer_norm")
    r11 = re.compile(r"pre_classifier")
    result = r1.sub(prefix_bert+'embeddings', l_name)
    result = r2.sub(prefix_bert+'encoder', result)
    result = r3.sub('self.query', result)
    result = r4.sub('self.value', result)
    result = r5.sub('self.key', result)
    result = r6.sub('output.dense', result)
    result = r7.sub('attention.output.LayerNorm', result)
    result = r8.sub('intermediate.dense', result)
    result = r9.sub('output.dense', result)
    result = r10.sub('output.LayerNorm', result)
    result = r11.sub(prefix_bert+'pooler.dense', result)
    if result in model_bert.state_dict().keys():
        #print(l_name+' -> '+result)
        state_dict_target[result] = model_dist.state_dict()[l_name]
        n_matches += 1
token_type_embeddings = model_bert.state_dict()[prefix_bert+'embeddings.token_type_embeddings.weight']
state_dict_target[prefix_bert+'embeddings.token_type_embeddings.weight'] = torch.zeros(token_type_embeddings.size())
print('Number of matches: ')
print(n_matches)
model_bert.load_state_dict(state_dict_target)
model_bert.eval()
model_dist.eval()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
outputs_dist = model_dist(input_ids)
outputs_bert = model_bert(input_ids)
last_hidden_states_dist = outputs_dist[0]
last_hidden_states_bert = outputs_bert[0]
print(last_hidden_states_dist)
print(last_hidden_states_bert)


