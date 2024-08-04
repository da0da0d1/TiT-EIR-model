import pickle
import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataAug import Exchange_Block, Concat_Prior_to_Last
from dataloader import EViTPair
from ..data_utils import split_data
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ..schedule import  WarmupMultiStepLR
from model import supervised_net
import numpy as np
from ..unsupervised_model.model import CLIP
from loss import RankedLoss, CrossEntropyLabelSmooth
import pickle

parser = argparse.ArgumentParser(description='Train supervised on EViT')
args = parser.parse_args('')  # running in ipynb
# set command line arguments here when running in ipynb

args.lr = 1e-4
args.weight_decay = 5e-5
args.schedule = []

args.type = 'Corel10K'
args.epochs = 70

# Fine-Turning
model = CLIP().cuda()
model.load_state_dict(torch.load('unsupervised_'+args.type+'_model_last.pth')['state_dict'])

# data transform
train_data,train_data_b, test_data,test_data_b, train_huffman_feature,train_huffman_feature_b,test_huffman_feature,test_huffman_feature_b, train_label, test_label = split_data(type=args.type)

train_transform = transforms.Compose([
    Exchange_Block(0.3),
    Concat_Prior_to_Last(0.3),
    transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])


train_data_EVIT = EViTPair(img_data=train_data, img_data_b=train_data_b, huffman_feature=train_huffman_feature,huffman_feature_b=train_huffman_feature_b,labels=train_label, transform=train_transform)
train_loader = DataLoader(train_data_EVIT, batch_size=20, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

test_data_EVIT = EViTPair(img_data=test_data,img_data_b=test_data_b,huffman_feature=test_huffman_feature,huffman_feature_b=test_huffman_feature_b,labels=test_label, transform=test_transform)
test_loader = DataLoader(test_data_EVIT, batch_size=20, shuffle=False, num_workers=20, pin_memory=True)


# loss
ce = CrossEntropyLabelSmooth()
rankloss = RankedLoss()


# train for one epoch
def train(net, data_loader, train_optimizer, scheduler, epoch, args):
    net.train()
    scheduler.step()

    total_loss, total_num, correct_num1, correct_num2, train_bar = 0.0, 0, 0, 0, tqdm(data_loader)

    for im_1, im_1_b, huffman, huffman_b, label in train_bar:
        im_1, im_1_b, huffman, huffman_b, label = im_1.cuda(non_blocking=True), im_1_b.cuda(
            non_blocking=True), huffman.cuda(non_blocking=True), huffman_b.cuda(non_blocking=True), label.cuda(non_blocking=True)

        fea1, out1, fea2, out2 = net(im_1, im_1_b, huffman, huffman_b, label)
        loss1 = ce(out1, label) + rankloss(fea1, label)
        loss2 = ce(out2, label) + rankloss(fea2, label)
        loss = (loss1 + loss2) / 2

        # 计算准确率
        correct_num1 += (out1.argmax(dim=1) == label).sum().item()
        correct_num2 += (out2.argmax(dim=1) == label).sum().item()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += label.size(0)
        total_loss += loss1.item() * label.size(0)

        accuracy1 = correct_num1 / total_num
        accuracy2 = correct_num2 / total_num

        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Acc1: {:.4f}, Acc2: {:.4f}'.format(epoch + 1, args.epochs,
                                                                                      train_optimizer.param_groups[0]['lr'],
                                                                                      total_loss / total_num, accuracy1, accuracy2))

    return total_loss / total_num, accuracy1, accuracy2

## test stage
def test(net, test_loader, test_label):
    net.eval()
    feature_bank = []
    train_bar = tqdm(test_loader)
    with torch.no_grad():
        for im_1, im_1_b, huffman, huffman_b in train_bar:
            im_1, im_1_b, huffman, huffman_b = im_1.cuda(non_blocking=True), im_1_b.cuda(
                non_blocking=True), huffman.cuda(non_blocking=True), huffman_b.cuda(non_blocking=True)

            feature1, feature2 = net(im_1, im_1_b, huffman, huffman_b)
            feature1 = F.normalize(feature1, dim=1)
            feature2 = F.normalize(feature2, dim=1)
            feature = 0.6*feature1 + 0.4*feature2
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        feature_labels = torch.tensor(test_label, device=feature_bank.device)
        average_precision_li = []
        for idx in range(feature_bank.size(0)):
            query = feature_bank[idx].expand(feature_bank.shape)

            label = feature_labels[idx]
            sim = F.cosine_similarity(feature_bank, query)
            _, indices = torch.topk(sim, 100)
            match_list = feature_labels[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []

            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if precision_li == []:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
        print('image test mAP:',mAP)


if __name__ == '__main__':

    S_net = supervised_net(net=model.net, out_dim=100)
    S_net.cuda()


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=[60])

    now_time = datetime.datetime.now()
    print(now_time)

    epoch_start = 0

    ##training loop
    supervised_loss_values = []
    supervised_accuracy1 = []
    supervised_accuracy2 = []
    for epoch in range(epoch_start, args.epochs):
       train_loss, accuracy1,accuracy2 = train(S_net, train_loader, optimizer,scheduler, epoch, args)
       supervised_loss_values.append(train_loss)
       supervised_accuracy1.append(accuracy1)
       supervised_accuracy2.append(accuracy2)

    with open('supervised_loss_value.pkl', 'wb') as file:
       pickle.dump(supervised_loss_values, file)
    with open('image_supervised_accuracy.pkl', 'wb') as file:
       pickle.dump(supervised_accuracy1, file)
    with open('text_supervised_accuracy.pkl', 'wb') as file:
       pickle.dump(supervised_accuracy2, file)

    torch.save({'epoch': epoch, 'state_dict': S_net.state_dict(), 'optimizer' : optimizer.state_dict(),}, 'supervised_'+args.type+'_model_last.pth')

    now_time = datetime.datetime.now()
    print(now_time)

    test(S_net, test_loader, test_label)

    now_time = datetime.datetime.now()
    print(now_time)
