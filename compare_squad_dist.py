from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer
)
from torch.utils.data import (DataLoader, SequentialSampler)
import re
import torch
import sys
# Change for sequence classification model
from model_bert_similar_dist import BertModel, BertForQuestionAnswering

MODEL_CLASSES  = {
    "bert" : (
        BertConfig,
        BertForQuestionAnswering,
        BertTokenizer,
    ),
    "distilbert":(
        DistilBertConfig,
        DistilBertForQuestionAnswering,
        DistilBertTokenizer,
    )
}
# Initialize config classes
config_class_target, model_class_target, tokenizer_class_target = MODEL_CLASSES['bert']
config_class_source, model_class_source, tokenizer_class_source = MODEL_CLASSES['distilbert']
config_bert = config_class_target.from_pretrained('./output_bert_squad')
config_dist = config_class_source.from_pretrained('./output_dist_squad')
# Save config
#config_dist.save_pretrained('./output_dist_squad')
#config_bert.save_pretrained('./output_bert_squad')
#sys.exit(0)
# Initialize models
model_dist = model_class_source.from_pretrained('distilbert-base-cased-distilled-squad')
model_bert = model_class_target(config_bert)
# Initialize tokenizers
tokenizer = tokenizer_class_source.from_pretrained('distilbert-base-cased-distilled-squad')
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
    else:
        print(result)
token_type_embeddings = model_bert.state_dict()[prefix_bert+'embeddings.token_type_embeddings.weight']
state_dict_target[prefix_bert+'embeddings.token_type_embeddings.weight'] = torch.zeros(token_type_embeddings.size())
print('Number of matches: ')
print(n_matches)
model_bert.load_state_dict(state_dict_target)
#model_bert.eval()
#model_dist.eval()
#input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
#outputs_dist = model_dist(input_ids)
#outputs_bert = model_bert(input_ids)
#last_hidden_states_dist = outputs_dist[0]
#last_hidden_states_bert = outputs_bert[0]
#print(last_hidden_states_dist)
#print(last_hidden_states_bert)
# Install a pip package in the current Jupyter kernel
import sys


import os

# Create a directory to store predict file
output_dir = "./pytorch_output"
cache_dir = "./pytorch_squad"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
# create cache dir
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Download the file
predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
if not os.path.exists(predict_file):
    import wget
    print("Start downloading predict file.")
    wget.download(predict_file_url, predict_file)
    print("Predict file downloaded.")


# Specify some model config variables.


# Define some variables
model_type = "bert"
model_name_or_path = "bert-base-cased"
max_seq_length = 128
doc_stride = 128
max_query_length = 64
per_gpu_eval_batch_size = 1
eval_batch_size = 1
import torch
device = torch.device("cpu")

from transformers.data.processors.squad import SquadV2Processor
processor = SquadV2Processor()
examples = processor.get_dev_examples(None, filename=predict_file)

from transformers import squad_convert_examples_to_features
features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset='pt'
        )

cached_features_file = os.path.join(cache_dir, 'cached_{}_{}_{}'.format(
        'dev',
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(384))
    )

torch.save({"features": features, "dataset": dataset}, cached_features_file)
print("Saved features into cached file ", cached_features_file)

# create output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_gpu = torch.cuda.device_count()
# eval_batch_size = 8 * max(1, n_gpu)

eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

# multi-gpu evaluate


# ## 2. Export the loaded model ##
# Once the model is loaded, we can export the loaded PyTorch model to ONNX.

# Eval!
print("***** Running evaluation {} *****")
print("  Num examples = ", len(dataset))
print("  Batch size = ", eval_batch_size)

output_model_path = './pytorch_squad/bert-base-cased-squad.onnx'
inputs = {}
outputs= {}
# Get the first batch of data to run the model and export it to ONNX
batch = dataset[0]

# Set model to inference mode, which is required before exporting the model because some operators behave differently in
# inference and training mode.
model_dist.eval()
batch = tuple(t.to(device) for t in batch)
inputs = {
    'input_ids':      batch[0].reshape(1, 128),                         # using batch size = 1 here. Adjust as needed.
    'attention_mask': batch[1].reshape(1, 128)#,
    #'token_type_ids': batch[2].reshape(1, 128)
}
with torch.no_grad():
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model_dist,                                            # model being run
                      (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                       inputs['attention_mask']),#,
                       #inputs['token_type_ids']),
                      output_model_path,                                # where to save the model (can be a file or file-like object)
                      opset_version=11,                                 # the ONNX version to export the model to
                      do_constant_folding=True,                         # whether to execute constant folding for optimization
                      input_names=['input_ids',                         # the model's input names
                                   'input_mask'],
                                   #'segment_ids'],
                      output_names=['start', 'end'],                    # the model's output names
                      dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                    'input_mask' : symbolic_names,
                                    #'segment_ids' : symbolic_names,
                                    'start' : symbolic_names,
                                    'end' : symbolic_names})
    print("Model exported at ", output_model_path)


# ## 3. Inference the Exported Model with ONNX Runtime ##
#
# #### Install ONNX Runtime
# Install ONNX Runtime if you haven't done so already.
#
# Install `onnxruntime` to use CPU features, or `onnxruntime-gpu` to use GPU.


import onnxruntime as rt
import time
import psutil

sess_options = rt.SessionOptions()

# Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization and store the optimized graph to desired location.
sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model.onnx")
sess_options.intra_op_num_threads=1
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'PASSIVE'
session = rt.InferenceSession(output_model_path, sess_options)

# evaluate the model
start = time.time()
res = session.run(None, {
          'input_ids': inputs['input_ids'].cpu().numpy(),
          'input_mask': inputs['attention_mask'].cpu().numpy()#,
          #'segment_ids': inputs['token_type_ids'].cpu().numpy()
        })
end = time.time()
print("ONNX Runtime inference time: ", end - start)


# Get perf numbers from the original PyTorch model.

start = time.time()
outputs = model_dist(**inputs)
end = time.time()
print("PyTorch Inference time = ", end - start)
print("***** Verifying correctness *****")
import numpy as np
for i in range(2):
    print('PyTorch and ORT matching numbers:', np.allclose(res[i], outputs[i].cpu().detach().numpy(), rtol=1e-04, atol=1e-05))
