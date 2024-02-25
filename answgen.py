import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
MAX_LENGTH=128
DEVICE='cpu'
final_dataset = pd.read_csv('final_dataset_homer.csv')
base_answers = final_dataset['A']
all_answers = list(set(base_answers)) # Список всех ответов из базы
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.from_pretrained("bi_encoder_homer")

class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)
model = CrossEncoderBert().to(DEVICE)
model.bert_model.from_pretrained("cross_encoder_homer")

def get_answear(
    tokenizer, finetuned_ce, base_bert, query, context, corpus,
    size_patch=150, qty_rand_choose=4, max_out_context=200
):

    # Создаем словарь для хранения оценок и ответов
    dic_answear = {"score": [], "answer": []}

    # Объединяем запрос и контекст памяти
    context_memory = query + "[SEP]" + context

    # Ограничиваем количество случайно выбираемых ответов
    if len(corpus) < qty_rand_choose * max_out_context:
        qty_rand_choose = int(len(corpus))

    # Поскольку база большая, проводим несколько выборов случайных ответов
    for i in range(qty_rand_choose):
        rand_patch_corpus = list(np.random.choice(corpus, size_patch))

        # Токенизируем запросы и случайно выбранные ответы
        queries = [context_memory] * len(rand_patch_corpus)
        tokenized_texts = tokenizer(
            queries, rand_patch_corpus, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)

        # Оцениваем модель Finetuned CrossEncoder
        with torch.no_grad():
            ce_scores = finetuned_ce(tokenized_texts['input_ids'], tokenized_texts['attention_mask']).squeeze(-1)
            ce_scores = torch.sigmoid(ce_scores)  # Применяем сигмоиду при необходимости

        # Обрабатываем оценки для модели Finetuned
        scores = ce_scores.cpu().numpy()
        scores_ix = np.argsort(scores)[::-1][0]
        dic_answear["score"].append(scores[scores_ix])
        dic_answear["answer"].append(rand_patch_corpus[scores_ix])

    # Находим наилучший ответ и его оценку
    best_answer_index = np.argsort(dic_answear["score"])[::-1][0]
    best_answer = dic_answear["answer"][best_answer_index]

    # Обновляем контекст памяти
    conext_memory = best_answer + "[SEP]" + context_memory
    return best_answer, conext_memory[:max_out_context]

def answer(question, context):
    answer,_ = get_answear(
                tokenizer, model, bert_model.to(DEVICE),
                query = question,
                context = context,
                corpus = all_answers)
    return answer