
import numpy as np

import torch
import torch.nn.functional as F
from transformers.optimization import SchedulerType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from huggingface_hub import login
from pyvi import ViTokenizer, ViPosTagger
from rank_bm25 import BM25Okapi

from .process_data import split_sentence, preprocess_text
def select_sentance_text(context, claim, thres= 0.6):
    
    tfidf_vectorizer = TfidfVectorizer()
    corpus = split_sentence(context)
    answer = corpus.copy()
    claim = preprocess_text(claim)
    claim = ViTokenizer.tokenize(claim).lower()
    
    len_claim = len(claim.split(' '))
    
    corpus_pro = []
    
    for i in range(len(corpus)):
        corpus[i] = preprocess_text(corpus[i])
        corpus[i] = ViTokenizer.tokenize(corpus[i]).lower()
        
        sentence = corpus[i]
        
        l = len(sentence.split(' '))
        
        p  = l/len_claim
        
        if i != 0 and p < thres and l > 1 :
            sentence = f'{corpus[i-1]}. {sentence}'
        corpus_pro.append(sentence)
    corpus_pro.append(claim)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_pro)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    score = []
    
    for i in range(len(cosine_sim)-1):
        score.append((cosine_sim[len(corpus_pro)-1, i], answer[i]))
    
    score = sorted(score, reverse=True)
    ans = []
    for s, sen in score[:1]:
        ans.append(sen)
    return score[0][1]
def find_nei_evi(context, claim, model_evidence_f1, tokenizer_f1):
    model_evidence_f1.eval()
    inputs = tokenizer_f1(
        claim, 
        context, 
        max_length = 512, 
        return_tensors="pt",
        truncation="only_second", 
        padding="max_length"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    pt, start_logits, end_logits = model_evidence_f1(input_ids = input_ids, attention_mask = attention_mask)

    start_logits = start_logits.detach().cpu().numpy()
    end_logits = end_logits.detach().cpu().numpy()
    
    # Lấy vị trí của token có giá trị logit lớn nhất cho vị trí bắt đầu và kết thúc
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)


    # Lấy các token tương ứng với vị trí bắt đầu và kết thúc trong văn bản
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]

    # Chuyển đổi token thành văn bản
    evidence = tokenizer_f1.decode(answer_tokens)

    ### check characte species
    if '</s></s>' in evidence :
        evidence = evidence.split('</s></s>')[1]
    # check number of sentence in evidence predict
    cntx = 0
    for p in split_sentence(evidence):
        if p.strip()!='':
            cntx+=1

    evidence = evidence.replace('<s>', '')
    evidence = evidence.replace('</s>', '')

    
    if evidence =='<s>' or len(evidence) == 0 or cntx > 1:
        return -1
    else:
        lines = split_sentence(context)
        for line in lines:
            if preprocess_text(evidence) in preprocess_text(line):
                return line
        print('error: not find evi in context')
        
        print(lines)
        print('==========')
        print(evidence)
        return evidence
def check_evidence(context, claim, model_evidence_f1, tokenizer_f1):
    lines = split_sentence(context)
    tokens = context.split(' ')
    
    if len(tokens) <= 400: 
        evi = find_nei_evi(claim = claim, context = context, model_evidence_f1 = model_evidence_f1, tokenizer_f1 = tokenizer_f1)
        if evi == -1:   
            return -1, select_sentance_text(context = context, claim = claim)
        return 0 , evi
    
        
    token_line = [l.split(' ') for l in lines]
    
    tmp_context_token = []
    tmp_context = []
    cnt = 0
    error = []
    evidence_list = []
    for idx in range(len(lines)):
        check = True
        if len(token_line[idx] + tmp_context_token) <=400:
            tmp_context_token += token_line[idx]
            tmp_context.append(lines[idx])
            check = False
        
        if len(token_line[idx] + tmp_context_token) > 400 or idx == len(lines) - 1:
            context_sub = '. '.join(tmp_context)
            if len(context_sub)== 0: 
                continue
            label = 'NEI'
            evidence = find_nei_evi(claim = claim, context = context_sub, model_evidence_f1 = model_evidence_f1, tokenizer_f1 = tokenizer_f1)

               
            if evidence != -1: # If appear evidence
                
                evidence_list.append(evidence)
            if check:
                tmp_context_token = token_line[idx] 
                tmp_context = [lines[idx]]
            else:
                tmp_context_token = []
                tmp_context = []

    if len(evidence_list) == 1: 
        return 0, evidence_list[0]
    else :
        if len(evidence_list) == 0: 
            return -1, select_sentance_text(context = context, claim = claim)
        else: 
            return 0, select_sentance_text(context = context, claim = claim)
def classify_nei(claim, evidence, model, tokenazation):
    
    model.eval()

    context_sub =evidence
    claim_sub = claim

    encoding = tokenazation(
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

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_masks']

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,

        )
    outputs = F.softmax(outputs, dim=1)
    _, pred = torch.max(outputs, dim=1)

    return  _, pred.item()
def get_top_context(context, claim = None, topk = None):
    context = split_sentence(context)
    if topk == None:
        return context
    
    
    context = [line for line in context]

    tokenized_context = [doc.split(' ') for doc in context]
    bm25 = BM25Okapi(tokenized_context)
    scores = bm25.get_scores(claim.split())

    score_sentence_pairs = sorted(zip(scores, context), reverse=True)
    highest_sentence = []

    for _, x in score_sentence_pairs[:topk]:

        highest_sentence.append(x)
        
    return highest_sentence