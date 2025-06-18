# ======================== libraries ========================

# 문서 작성
from docx import Document
from docx.shared import Pt,RGBColor

# 데이터 처리
import pandas as pd
import numpy as np
import re
import json
import pickle

# HTTP & XML 파싱
import requests
import xmltodict

# 자연어 처리
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from okt import build_bag_of_words

# PyTorch
import torch
import torch.nn as nn

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, BertPreTrainedModel, BertModel

# Fine-Tuning
from peft import PeftModel, PeftConfig

# 분류
from keras.preprocessing.sequence import pad_sequences

# ======================== EDIT ========================

# 최종 파일 이름
FILE_NAME = "NCCOSS_KOSPAI.docx"

# keyword.csv 경로
KEYWORD_CSV = '/home/in2wise/economy/keyword.csv'

# stopwords.json 경로
STOPWORDS_JSON = '/home/in2wise/NCCOSS/stopwords.json'

# 키프리스 REST KEY
RESTkey = ""

# 기술분야 작성 모델 경로 (mistral)
MODEL_TECH_FIELD = "/home/in2wise/mis_기술분야_epochs15"

# 배경기술 작성 모델 경로
MODEL_BG_ART = "/home/in2wise/배경기술_fin"

# 해결하려는 과제 작성 모델 경로 (코알파카)
MODEL_ASSIGN = "/home/in2wise/economy/KoAlpaca_vol13_save"

# 과제 해결 수단 작성 모델 경로
MODEL_SOLVE = "/home/in2wise/solution_epochs20"

# 요약 모델 경로 (t5-large)
MODEL_SUMMARY = "lcw99/t5-large-korean-text-summary"

# 분류 관련 경로
MODEL_CLASSIFICATION = '/home/in2wise/classification/MultiLabel/model/bert_D'
LE = '/home/in2wise/classification/MultiLabel/tensor_D/labels_le.pkl'

# ======================== basic settings ========================

# 발명신고서 내용: 사용자 입력 받기
TITLE = input("\n\n발명의 명칭 : ")
STRUCTURE = input("\n\n발명의 구성 : ")
EFFECT = input("\n\n발명의 효과 : ")

# 문서 설정
document = Document()
style = document.styles['Normal']
font = style.font
font.name = '나눔고딕'
font.size = Pt(10)
document_name = FILE_NAME

paragraph = document.add_paragraph()

# ======================== LLM functions ========================

def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    generated_text = tokenizer.decode(gened[0], skip_special_tokens=True)
    answer_start_index = generated_text.find("### 답변:") + len("### 답변:")
    generated_answer = generated_text[answer_start_index:].strip()
    return generated_answer

# 과제의 해결 수단 항목
def gen_solution(x):
    q = f"### 질문: {x}\n\n### 답변:"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=512,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    generated_text = tokenizer.decode(gened[0], skip_special_tokens=True)
    answer_start_index = generated_text.find("### 답변:") + len("### 답변:")
    generated_answer = generated_text[answer_start_index:].strip()
    return generated_answer

# ======================== Classification functions ========================

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        _, pooled_output = self.bert(input_ids, attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

def text_to_loader_bert(sentences, max_len):
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    
    sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    [sentence.insert(0,'[CLS]') for sentence in sentences]
    [sentence.append('[SEP]') if len(sentence)<max_len-1 else sentence.insert(max_len-1, '[SEP]') for sentence in sentences]
    
    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', padding='post', truncating='post')
    
    attention_masks = []
    
    for input_id in input_ids:
        attention_mask = [1 if (i>0) else 0 for i in input_id]
        attention_masks.append(attention_mask)
    
    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)
    
    return train_inputs, train_masks

# ======================== 발명의 명칭 ========================

paragraph.add_run(f"발명의 명칭 ")
text_name = re.sub(r"[^가-힣\s]", "", TITLE)
contents = paragraph.add_run(text_name)
contents.bold = True
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

# ======================== 선행기술문헌 ========================

contents = paragraph.add_run("[선행기술문헌]\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

# TF-IDF로 키워드 추출하기

df = pd.read_csv(KEYWORD_CSV, encoding='UTF8')
columns = ['발명의 명칭', 'keyword']

df = df[columns]
df = df.astype('str')
input_df = df.fillna('')
keyword_df = input_df[['keyword']]
lists = keyword_df['keyword'].values.tolist()
len_lists = len(lists)

with open(STOPWORDS_JSON, 'r') as f:
    stop_list = json.load(f)
f.close()

sentence = []
for words in TITLE.split():
    vocab, bow = build_bag_of_words(words)
    
    # 타이틀에서 불용어 제거
    filtered_words = []
    for word in vocab:
        if word in stop_list:
            continue
        filtered_words.append(word)
    
    # 불용어 제거 후 문장화
    sentence.append(''.join(filtered_words))
    
lists.append(' '.join(sentence))

tfidfv = TfidfVectorizer().fit(lists)
X_tfidf = tfidfv.fit_transform(lists)
feature_names = tfidfv.get_feature_names_out()
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)

# 상위 키워드 추출

key = []
for doc_index, doc_row in enumerate(X_tfidf):
    non_zero_elements = doc_row.nonzero()[1]
    if non_zero_elements.any():
        words = {}
        for word_index in non_zero_elements:
            word = list(tfidfv.vocabulary_.keys())[list(tfidfv.vocabulary_.values()).index(word_index)]
            words[word] = doc_row[0, word_index]
        words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        tmp = []
        
        # 각 문장에 대해 TF-IDF가 높은 단어 최대 2개 추출
        for k in range(len(words) if len(words) < 2 else 2):
            tmp.append(words[k][0])
    key.append(' '.join(tmp))

# 추출한 키워드를 키프리스에서 검색해 선행 기술 문헌 작성

url1 = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch?inventionTitle="
url2 = "&ServiceKey="

key = key[len_lists].replace(' ', '*')

response = requests.get(url1 + key + url2 + RESTkey)
content = response.content
dict_type = xmltodict.parse(content)

# 최종 검색 결과
items = dict_type['response']['body']['items']

final = ""

# 결과가 없음
if items is None:
    final = "\n"

# 하나 존재
elif isinstance(items['item'], dict):
    items_list = [items['item']]

# 3개만 추출
elif isinstance(items['item'], list):
    items_list = items['item'][:3]

# 예외 (오류 등)
else:
    item_list = []

for idx, item in enumerate(items_list, start=1):
    num = item['applicationNumber']
    title = item['inventionTitle']
    line = f"(특허문헌 000{idx}) 한국등록특허 제10-{num[2:6]}-{num[6:]}호 <{title}>"
    final += line + ("\n" if idx < len(items_list) else "")

paragraph.add_run(final)
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

print('\n\n\n선행기술문헌 작성 완료\n\n\n')

# ======================== 기술분야 ========================

contents = paragraph.add_run("[기 술 분 야]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

model_id = MODEL_TECH_FIELD
config = PeftConfig.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config = bnb_config,
    device_map = {"": 0}
)

model = PeftModel.from_pretrained(model, model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()

paragraph.add_run(gen(TITLE))
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

print('\n\n\n기술분야 작성 완료\n\n\n')

# ======================== 배경기술 ========================

contents = paragraph.add_run("[배 경 기 술]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

model_id = MODEL_BG_ART
config = PeftConfig.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config = bnb_config,
    device_map = {"": 0}
)

model = PeftModel.from_pretrained(model, model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()

paragraph.add_run(gen(TITLE))
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')
    
print('\n\n\n배경기술 작성 완료\n\n\n')

# ======================== 청구범위 ========================

contents = paragraph.add_run("[청 구 범 위]\n\n")
contents.font.color.rgb = RGBColor(255,0,127)
contents.bold = True
contents.italic = True

paragraph.add_run(STRUCTURE)
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

# ======================== 해결하려는 과제 ========================

contents = paragraph.add_run("[해결하려는 과제]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

model_id = MODEL_ASSIGN
config = PeftConfig.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config = bnb_config,
    device_map = {"": 0}
)

model = PeftModel.from_pretrained(model, model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()

paragraph.add_run(gen(TITLE))
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

print('\n\n\n해결하려는 과제 작성 완료\n\n\n')

# ======================== 과제 해결 수단 ========================

contents = paragraph.add_run("[과제 해결 수단]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

model_id = MODEL_SOLVE
config = PeftConfig.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config = bnb_config,
    device_map = {"": 0}
)

model = PeftModel.from_pretrained(model, model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.eval()

paragraph.add_run(gen_solution(STRUCTURE))
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

print('\n\n\n과제의 해결 수단 작성 완료\n\n\n')

# ======================== 발명의 효과 ========================

contents = paragraph.add_run("[발명의 효과]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

paragraph.add_run(EFFECT)
paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

# ======================== 요약 ========================

contents = paragraph.add_run("[요약]\n\n")
contents.font.color.rgb = RGBColor(255, 0, 127)
contents.bold = True
contents.italic = True

nltk.download('punkt')

model_dir = MODEL_SUMMARY
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512 + 256

inputs = [gen(TITLE)]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
paragraph.add_run(predicted_title)

inputs = [EFFECT]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
paragraph.add_run(predicted_title)

inputs = [gen_solution(STRUCTURE)]
inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
paragraph.add_run(predicted_title)

paragraph.add_run('\n')

paragraph.add_run('_' * 79)
paragraph.add_run('\n')

print('\n\n\n요약 작성 완료')

# ======================== 분류 ========================

contents = paragraph.add_run("[분류]\n\n")
contents.font.color.rgb = RGBColor(255,0,127)
contents.bold = True
contents.italic = True

test_input, test_mask = text_to_loader_bert(STRUCTURE, 512)
test_input.to("cuda")
test_mask.to("cuda")

model_id = MODEL_CLASSIFICATION

model = BertForMultiLabelSequenceClassification.from_pretrained(model_id, cache_dir=None, num_labels=8)
model = model.cuda()

with open(LE, 'rb') as f:
    le = pickle.load(f)

arr1 = []
arr2 = []

test = model(test_input[0:1].to("cuda"), token_type_ids=None, attention_mask=test_mask[0:1].to("cuda"))
values, indexs = torch.topk(test,2)

for index in indexs:
    arr1.append(index[0].item())
    arr2.append(index[1].item())

result1 = le.inverse_transform(arr1)
result2 = le.inverse_transform(arr2)

result1 = ''.join(list(result1[0]))
result2 = ''.join(list(result2[0]))

paragraph.add_run(result1)
paragraph.add_run('\n')
paragraph.add_run(result2)

print('\n\n\n\n특허명세서 작성 완료')

document.save(document_name)
