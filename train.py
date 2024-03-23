# coding:utf-8
"""
# @Time    : 2024.01.10
# @Author  : Xinglin Lian
# @Contact : kenshin_lian@qq.com
# @Description : Training scripts for Lenovo Sensitive Phrase Detection, based on a large language model with supervised classification
# @Software: Win 10 or linux
"""

from tqdm import tqdm
from dataset import GenDateSet
import os, argparse, time, datetime
import torch
from torch import nn
from transformers import AutoConfig
from torch.nn import CrossEntropyLoss
from model import ElectraForPairwiseCLS, MinirbtForPairwiseCLS

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Net para search of corresponding documents')
    # run way
    parser.add_argument('--run_way', type=str, default='model_train', help="options: ['vec_match', 'model_train'], code run way")
    # model choose
    parser.add_argument('--model_name', type=str, default='minirbt', help="options: ['Electra', 'minirbt'], ")
    parser.add_argument('--model_dir', type=str, default='/home/gpu/PycharmProjects/Pretrain_model_test/Pretrain_week_2/sensitive_project/minirbt-h256/model', help="Pre-trained model path")

    parser.add_argument('--model_save_path', type=str, default='/home/gpu/PycharmProjects/Pretrain_model_test/Pretrain_week_2/sensitive_project/minirbt-h256', help="Trained model save path")
    # dataset
    parser.add_argument('--trainSet_path', type=str, default='/home/gpu/PycharmProjects/Pretrain_model_test/Pretrain_week_2/sensitive_project/data/data_3.19/train_data.csv', help="Train dataset path")
    parser.add_argument('--valSet_path', type=str, default='/home/gpu/PycharmProjects/Pretrain_model_test/Pretrain_week_2/sensitive_project/data/data_3.19/validation_data.csv', help="Validation dataset path")
    parser.add_argument('--testSample_path', type=str, default='', help="Test samples path")
    parser.add_argument('--sensiDB_path', type=str, default='', help="Sensitive words database path")
    # hyperparameter
    parser.add_argument('--max_length', type=int, default=128, help="Maximum length of each sentence")
    parser.add_argument('--num_classes', type=int, default=2, help="Model output classification type")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--epoch', type=int, default=30, help="Epoch for training, 30")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for training")
    parser.add_argument('--patience', type=int, default=3, help="Stopping step for early stopping in training")

    args= parser.parse_args()
    return args

def format_time(elapsed) -> str:
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_epoch(epoch_index, optimizer, model, device, train_data) -> str:
    model.train()
    batch_epoch = 0
    loss_fn = CrossEntropyLoss()
    print('\n======== Epoch {:} ========'.format(epoch_index + 1))
    t0 = time.time()
    for (input_id, attention_mask, label) in tqdm(train_data):
        input_id, attention_mask, label = input_id.to(device), attention_mask.to(device), label.to(device)

        outputs = model(input_id, attention_mask=attention_mask)  # forward
        optimizer.zero_grad()
        loss = loss_fn(outputs, label)
        loss.backward() # backward
        optimizer.step()    # update weight
        batch_epoch += 1
        if batch_epoch % 1000 == 0:
            print('Train Epoch:', epoch_index+1, ' , batch_epoch: ', batch_epoch, ' , loss = ', loss.item())
    return format_time(time.time() - t0)


def validate_epoch(model, device, data) -> float:
    model.eval()
    test_loss = 0.0
    acc = 0
    for (input_id, masks, label) in tqdm(data):
        input_id, masks, label = input_id.to(device), masks.to(device), label.to(device)
        with torch.no_grad():

            logits = model(input_id, attention_mask=masks)
        
        print(logits.shape)
        print(label.shape)

        test_loss += nn.functional.cross_entropy(logits, label.squeeze())
        pred = logits.max(-1, keepdim=True)[1]
        acc += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(data)
    return acc / len(data.dataset)


def train(args: argparse.Namespace) -> None:
    # load tokenizer and model
    tokenizer: None
    model: None
    if args.model_name == 'Electra':
        from transformers import ElectraTokenizer
        tokenizer = ElectraTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)
        # model = ElectraForPairwiseCLS.from_pretrained(args.model_dir, config=config)
        model = ElectraForPairwiseCLS(args.model_dir, config=config)
        print(model)

    elif args.model_name == 'minirbt':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)
        model = MinirbtForPairwiseCLS.from_pretrained(args.model_dir, config=config)
        print(model)

    else:
        raise ValueError('has no this model!')

    # load dataset
    train_dataset = GenDateSet(tokenizer, args.trainSet_path, args.max_length, args.batch_size)
    val_dataset = GenDateSet(tokenizer, args.valSet_path, args.max_length, args.batch_size)
    train_data = train_dataset.gen_data()
    val_data= val_dataset.gen_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model = model.to(device)
    print(f'<==== Training uses {device} =====>')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = CrossEntropyLoss()

    total_t0 = time.time()
    best_acc = 0.0
    no_improvement_count = 0
    model_save_path = os.path.join(args.model_save_path, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    if os.path.exists(model_save_path) == False:
        os.makedirs(model_save_path)

    model_save_path = os.path.join(model_save_path, 'output_model.pth')

    for epoch_index in range(args.epoch):

        # train and validation
        training_time = train_epoch(epoch_index=epoch_index, optimizer=optimizer, model=model, device=device, train_data=train_data)
        acc = validate_epoch(model=model, device=device, data=val_data)
        print(f'Training spend {training_time} h  ===>  Validation Acc = {acc}')

        # save the model if accuracy improves
        if acc > best_acc:
            # torch.save(model.state_dict(), args.model_save_path)
            torch.save(model, model_save_path)

            best_acc = acc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.patience:
                print(f'No improvement in validation accuracy for {args.patience} epochs. Early stopping.')
                break
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def vector_match(args: argparse.Namespace) -> None:
    """ Load tokenizer and model
    """
    tokenizer: None
    model: None
    if args.model_name == 'Electra':
        from transformers import ElectraTokenizer, ElectraForSequenceClassification
        tokenizer = ElectraTokenizer.from_pretrained(args.model_dir)
        model = ElectraForSequenceClassification.from_pretrained(args.model_dir, num_labels=args.num_classes)
    elif args.model_name == '':
        pass
    else:
        raise ValueError('has no this model!')

    # load testset and sensitive db
    test_dataset = GenDateSet(tokenizer, args.testSample_path, args.max_length, 1)
    sensi_db = GenDateSet(tokenizer, args.sensiDB_path, args.max_length, 1)
    test_data = test_dataset.gen_data()
    sensi_db = sensi_db.gen_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'<==== Testing uses {device} =====>')

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model = model.to(device)

    for (input_id, attention_mask, label) in tqdm(test_data):
        input_id, attention_mask, label = input_id.to(device), attention_mask.to(device), label.to(device)
        test_vec = model.electra(input_id, attention_mask=attention_mask)['last_hidden_state'][0, 0, :] # [batch size, word length, vec dimension]

        temp_max = 0
        for (input_id_sensi, attention_mask_sensi, label_sensi) in tqdm(sensi_db):
            input_id_sensi, attention_mask_sensi, label_sensi = input_id_sensi.to(device), attention_mask_sensi.to(device), label_sensi.to(device)
            vec_similar = torch.cosine_similarity(test_vec, model.electra(input_id_sensi, attention_mask=attention_mask_sensi)['last_hidden_state'][0, 0, :], dim=0)
            temp_max = vec_similar if vec_similar > temp_max else temp_max

        print(f'max similarity is : {temp_max.item()}')


if __name__ == '__main__':
    args = parse_arg()

    if args.run_way == 'vec_match':
        vector_match()
    elif args.run_way == 'model_train':
        train(args)
    else:
        raise ValueError('Error run way!')