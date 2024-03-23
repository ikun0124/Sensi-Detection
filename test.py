# coding:utf-8
"""
# @Time    : 2024.01.10
# @Author  : Xinglin Lian
# @Contact : kenshin_lian@qq.com
# @Description : Testing scripts for Lenovo Sensitive Phrase Detection, based on a large language model
# @Software: Win 10 or linux
"""

import psutil
import torch
from torch import nn
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, recall_score, precision_score
import argparse
import pandas as pd
import re
from transformers import ElectraTokenizer,BertTokenizer
import os
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Net para search of corresponding documents')
    # run way
    parser.add_argument('--run_way', type=str, default='model_for_sensitive', help="options: ['model_for_sensitive'], code run way")
    # model choose chinese-electra
    parser.add_argument('--model_name', type=str, default='Electra', help="options: ['minirbt','Electra'], ")
    # model path
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--model_load_path', type=str, default='', help="Trained model save path")
    # testset
    parser.add_argument('--TestSet_path', type=str, default='', help="Test dataset path")
    # record path
    parser.add_argument('--record_path', type=str, default='', help="output_result_path")
    # hyperparameter
    parser.add_argument('--max_length', type=int, default=128, help="Maximum length of each sentence")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")

    args= parser.parse_args()
    return args


class GenDateSet():
    """Dataset
    """
    def __init__(self, tokenizer, file, max_length=128, batch_size=16, split='train'):
        self.file = file
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.split = split

    def gen_data(self):
        if not os.path.exists(self.file):
            raise Exception("no dataset")

        df = pd.read_csv(self.file)
        if df.empty:
            raise Exception("no dataset")

        input_ids = []
        attention_masks = []
        labels = []

        # process data
        for index, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            try:
                label = int(row['label'])  # convert string to int
            except ValueError:
                label = 1
            # encoding text by tokenizer
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'  # return PyTorch tensor
            )

            input_ids.append(encoding['input_ids'].squeeze())  # compression dimension
            attention_masks.append(encoding['attention_mask'].squeeze())  # compression dimension
            labels.append(label)

        # generate TensorDataset
        data_gen = TensorDataset(torch.stack(input_ids),
                                torch.stack(attention_masks),
                                torch.LongTensor(labels))
        # generate and shuffle DataLoader
        if self.split == 'train':
            sampler = RandomSampler(data_gen)
            dataloader = DataLoader(data_gen, sampler=sampler, batch_size=self.batch_size)
        elif self.split == 'test':
            dataloader = DataLoader(data_gen, batch_size=self.batch_size)

        return dataloader


def load_model(device):
    """Model load
    """
    model_path = args.model_load_path
    if args.model_name == 'minirbt' or 'Electra':
        model = torch.load(model_path,map_location=device)
        print(model)
    else:
        raise ValueError('has no this model!')

    model.eval()
    return model


def tokenizer_choose(model_name):
    if model_name == 'Electra':
        tokenizer = ElectraTokenizer.from_pretrained(args.model_dir)
    elif model_name == 'minirbt':
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    else:
        raise ValueError('has no this model!')
    return tokenizer


def test(model, device, data, record_path ):
    model.eval()
    tokenizer = tokenizer_choose(args.model_name)
    test_loss = 0.0
    acc = 0
    total_time = 0  # time spend
    total_memory = 0  # RAM spend
    total_memory_allocated = 0 # GPU spend
    predictions = []
    labels = []
    incorrect_samples = []  # save error sample
    for (input_id, masks, label) in tqdm(data):
        start_time = time.time()
        input_id, masks, label = input_id.to(device), masks.to(device), label.to(device)
        with torch.no_grad():
            logits = model(input_id, attention_mask=masks)
        test_loss += nn.functional.cross_entropy(logits, label.squeeze())
        pred = logits.max(-1, keepdim=True)[1]
        acc += pred.eq(label.view_as(pred)).sum().item()

        predictions.extend(pred.squeeze().tolist())
        labels.extend(label.squeeze().tolist())
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_ms = elapsed_time * 1000

        # Got GPU memory usage
        try:
            torch.cuda.memory_allocated(device) 
        except:
            memory_allocated = 0
        else:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # display with MB

        # Got RAM memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # display with MB
        total_time += elapsed_time_ms
        total_memory += memory_usage
        total_memory_allocated += memory_allocated

        
        # Judge prediction and save correctly save error sample
        correct_predictions = pred.eq(label.view_as(pred))
        for i, correct in enumerate(correct_predictions):
            if not correct:
                input_text = tokenizer.decode(input_id[i], skip_special_tokens=True)
                incorrect_samples.append({
                    '输入文本': input_text,
                    '真实标签': label[i].item(),
                    '预测标签': pred[i].item()
                })

    test_loss /= len(data)
    accuracy = acc / len(data.dataset)
    f1 = f1_score(labels,predictions,average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')

    avg_gpu_memory = total_memory_allocated / len(data)
    avg_cpu_memory = total_memory / len(data)
    avg_prediction_latency = total_time / len(data)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # Model parameter, converted to M

    with open(record_path, 'w', encoding='utf-8') as f:
        f.write('测试结果：\n')
        f.write(f'模型参数量: {total_params:.2f}M\n')
        f.write(f'准确率：{accuracy:.4f}\n')
        f.write(f'F1分数：{f1:.4f}\n')
        f.write(f'召回率：{recall:.4f}\n')
        f.write(f'精确度：{precision:.4f}\n')
        f.write(f'平均预测时延：{avg_prediction_latency:.2f}ms\n')
        f.write(f'平均显存使用：{avg_gpu_memory:.2f}M\n')
        f.write(f'平均内存使用：{avg_cpu_memory:.2f}M\n')

        for sample in incorrect_samples:
            f.write('输入文本: {}\n'.format(sample['输入文本']))
            f.write('真实标签: {}\n'.format(sample['真实标签']))
            f.write('预测标签: {}\n'.format(sample['预测标签']))
            f.write('---\n')


# only keep Chinese and Arabic numerals
def keep_chinese(text):
    pattern = re.compile(r'[^\u4e00-\u9fa50-9，,。.:：;；？?!！]')  # Regular expressions that match non-Chinese characters and numbers
    chinese_and_numbers_text = re.sub(pattern, '', text)  # 使用正则表达式替换非中文和数字为空字符串
    return chinese_and_numbers_text


def main(args: argparse.Namespace) -> None:

    tokenizer = tokenizer_choose(args.model_name)
    gpu = input("输入：1.gpu  2.cpu\n")
    if gpu == '1':
        device = torch.device("cuda")
    elif gpu == '2':
        device = torch.device("cpu")
    else:
        raise RuntimeError('device error')
    # load model
    model = load_model(device)
    model = model.to(device)

    flag = input("输入：1.导入测试集进行预测  2.输入句子进行单条预测，敏感/非敏感\n")
    if flag=='1':
        test_dataset=GenDateSet(tokenizer, args.TestSet_path, args.max_length, args.batch_size, split='test')
        test_data=test_dataset.gen_data()
        test(model,device,test_data,args.record_path)

    elif flag=='2':
        while True:
            text = input("\n请输入内容:\n")

            text = keep_chinese(text)
            print("过滤后text：{}".format(text))
            if not text or text == "":
                continue
            elif text == "q":
                break

            # predict
            start_time = time.time()
            encoded_input = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=args.max_length,return_tensors="pt")
            input_ids = encoded_input['input_ids'].to(device)
            token_type_ids = encoded_input['token_type_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)

            output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            pred = output.max(-1, keepdim=True)[1][0].item()
            end_time = time.time()

            elapsed_time = end_time - start_time
            elapsed_time_ms = elapsed_time * 1000
            if pred == 0:
                print('输入为无毒')
            elif pred == 1:
                print('拒绝回答')
            else:
                print('等待模型生成回答')
            print(f"预测时间：{elapsed_time_ms}ms")
    else:
        raise RuntimeError('Note your input ')


if __name__ == '__main__':
    args = parse_arg()
    main(args)