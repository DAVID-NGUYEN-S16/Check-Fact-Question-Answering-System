from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
import argparse
# from transformers.models.bartpho.tokenization_bartpho_fast import BartphoTokenizerFast
from transformers import AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time
from transformers.optimization import SchedulerType
import streamlit as st
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
from huggingface_hub import login
from pyvi import ViTokenizer, ViPosTagger
from model.models import ClaimVerification
from model.models import ModelQA
from util.process_data import split_sentence, preprocess_text, process_data
from util.function import get_top_context
from util.function import check_evidence, classify_nei
login(token='xxxxx')
device = "cpu"
# Kiểm tra xem đã load model chưa
@st.cache_resource  
def load_model():
    
    print("Loading model")
    device = "cpu"
    tokenizer_rs = AutoTokenizer.from_pretrained("MoritzLaurer/ernie-m-large-mnli-xnli")
    tokenizer_f1 = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-base")
    tokenizer_evidence = AutoTokenizer.from_pretrained("MoritzLaurer/ernie-m-large-mnli-xnli")
    tokenizer_3_class = AutoTokenizer.from_pretrained("MoritzLaurer/ernie-m-large-mnli-xnli")

    checkpoint_classify_3_class = torch.load(f"./weight_model/classify-3-class/best_acc.pth", map_location=torch.device('cpu'))
    model_classify_3_class = ClaimVerification(n_classes=3, name_model="MoritzLaurer/ernie-m-large-mnli-xnli")
    model_classify_3_class.load_state_dict(checkpoint_classify_3_class)

    checkpoint_evidence = torch.load(f"./weight_model/evidence-by-classify/best_acc.pth", map_location=torch.device('cpu'))
    model_evidence = ClaimVerification(n_classes=2, name_model="MoritzLaurer/ernie-m-large-mnli-xnli")
    model_evidence.load_state_dict(checkpoint_evidence)

    checkpoint_evidence_f1 = torch.load(f"./weight_model/weght-model-base/best_model.pth", map_location=torch.device('cpu'))

    best_model_state_dict_evidence_f1 = checkpoint_evidence_f1['model_state_dict']
    model_evidence_f1 = ModelQA(name_model="nguyenvulebinh/vi-mrc-base")
    model_evidence_f1.load_state_dict(best_model_state_dict_evidence_f1)

    checkpoint_classify_rs = torch.load(f"./weight_model/classify-2-class/model_rs/best_acc.pth", map_location=torch.device('cpu'))
    model_classify_rs = ClaimVerification(n_classes=2, name_model="MoritzLaurer/ernie-m-large-mnli-xnli")
    model_classify_rs.load_state_dict(checkpoint_classify_rs)
        
    del checkpoint_evidence
    del checkpoint_classify_rs
    del checkpoint_classify_3_class
    del checkpoint_evidence_f1
    gc.collect()
    torch.cuda.empty_cache()

    return model_classify_3_class, model_evidence, model_evidence_f1, model_classify_rs, tokenizer_rs, tokenizer_f1,tokenizer_evidence, tokenizer_3_class

model_classify_3_class, model_evidence, model_evidence_f1, model_classify_rs, tokenizer_rs, tokenizer_f1,tokenizer_evidence, tokenizer_3_class  = load_model()

cag = ['NEI', 'SUPPORTED', 'REFUTED']

def infer(sample):
    start_time = time.time()
    
    context = process_data(sample['context'])
    claim = process_data(sample['claim'])
    submit = {}


    
    lines = get_top_context(context = context, claim = claim, topk=5)
    cnt = {0: 0, 1:0}
    evidence = ""

    
    for line in lines:
        encoding = tokenizer_evidence.encode_plus(
            claim,
            line,
            truncation="only_second",
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        inputs = {
                    'input_ids': encoding['input_ids'].reshape((1, 256)),
                    'attention_masks': encoding['attention_mask'].reshape((1, 256)),
                }

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_masks'].to(device)
        
        
        with torch.no_grad():
            outputs = model_evidence(
                input_ids=input_ids,
                attention_mask=attention_mask,

            )
        _, pred = torch.max(outputs, dim=1)
        cnt[pred[0].item()]+=1
        
        if pred[0].item() == 1:
            evidence = line
    if cnt[1] == 1:
        submit = {
                            'verdict': '1',
                            'evidence': evidence
        }
    else:
        
        not_nei, evidence = check_evidence(context = context, claim = claim, model_evidence_f1 = model_evidence_f1, tokenizer_f1 = tokenizer_f1)

        submit = {
                            'verdict': '3',
                            'evidence': evidence
        }
    
    ##### Classify #############
    
    model_classify_3_class.eval()

    context_sub = submit['evidence']
    claim_sub = claim
    encoding = tokenizer_3_class(
            claim_sub,
            context_sub,
            truncation="only_second",
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

    inputs = {
                'input_ids': encoding['input_ids'].reshape((1, 256)),
                'attention_masks': encoding['attention_mask'].reshape((1, 256)),
            }

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_masks'].to(device)

    with torch.no_grad():
        outputs = model_classify_3_class(
            input_ids=input_ids,
            attention_mask=attention_mask,

        )
    outputs = F.softmax(outputs, dim=1)

    prob3class, pred = torch.max(outputs, dim=1)
    
    if pred.item() == 0:
        submit['verdict'] ='NEI'
        submit['evidence'] =''
    else:
        prob2class, output_rs = classify_nei(claim = claim, evidence = submit['evidence'], model = model_classify_rs, tokenazation = tokenizer_rs)
        label_3class = cag[pred.item()]
        label_2class = ""
        if output_rs == 0:
            label_2class = 'SUPPORTED'
        else: 
            label_2class = 'REFUTED'
            
        submit['verdict'] = label_2class
        
        if label_3class != label_2class:
    
            if prob2class > prob3class:
                submit['verdict'] = label_2class
            else: submit['verdict'] = label_3class
        
    print(f"Time infer: {time.time() - start_time}")
    submit['time'] = time.time() - start_time
    return submit


## Streamlit app 

st.title('Hãy xác minh thông tin của bạn !!')

# Tạo input fields
context = st.text_area('Enter context:', 'Cậu ấy tên là Nam. Rất yêu cô ấy')
claim = st.text_area('Enter claim:', 'Cậu ấy tên là Hoàng')

# Tạo nút Submit
submit_button = st.button('Submit')

# Kiểm tra nút được bấm hay không
if submit_button:
    # Import hàm infer ở đây hoặc đặt hàm infer ngay trước đoạn này

    # Tạo inputs từ dữ liệu người dùng nhập
    inputs = {
        'claim': claim,
        'context': context
    }

    # Gọi hàm infer để nhận kết quả
    result = infer(inputs)

    # Hiển thị kết quả
    st.write('Với câu claim của bạn là:', claim)
    st.write('Chúng tôi tìm thấy được đây là một câu tuyên bố:', result['verdict'])
    st.write('Với minh chứng tìm được trong ngữ cảnh bạn cung cấp là:', result['evidence'])
    st.write('Time inference: ', result['time'])
