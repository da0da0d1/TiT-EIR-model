from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataAug import Exchange_Block, Concat_Prior_to_Last
from dataloader import EViTPair
from ..data_utils import split_data
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ..schedule import get_cosine_schedule_with_warmup
from model import CLIP
import numpy as np
import datetime
import pickle

parser = argparse.ArgumentParser(description='Train unsupervised on TiT')
args = parser.parse_args('')

## set training parameter
args.lr = 1e-3
args.weight_decay = 5e-5
args.type = 'Corel10K'
args.epochs = 300

train_data,train_data_b, test_data,test_data_b, train_huffman_feature,train_huffman_feature_b,test_huffman_feature,test_huffman_feature_b, train_label, test_label = split_data(type=args.type)

train_transform = transforms.Compose([
    Exchange_Block(0.3),
    Concat_Prior_to_Last(0.3),
    transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])

train_data_EVIT = EViTPair(img_data=train_data, img_data_b=train_data_b, huffman_feature=train_huffman_feature,huffman_feature_b=train_huffman_feature_b, transform=train_transform)
train_loader = DataLoader(train_data_EVIT, batch_size=20, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

test_data_EVIT = EViTPair(img_data=test_data,img_data_b=test_data_b,huffman_feature=test_huffman_feature,huffman_feature_b=test_huffman_feature_b, transform=test_transform)
test_loader = DataLoader(test_data_EVIT, batch_size=20, shuffle=False, num_workers=20, pin_memory=True)


# train one epoch
def train(net, data_loader, train_optimizer, epoch, scheduler, args):
    net.train()
    scheduler.step()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_1_b, huffman, huffman_b in train_bar:
        im_1, im_1_b, huffman, huffman_b = im_1.cuda(non_blocking=True), im_1_b.cuda(non_blocking=True),  huffman.cuda(non_blocking=True), huffman_b.cuda(non_blocking=True)

        loss = net(im_1, im_1_b, huffman, huffman_b)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch+1, args.epochs,
                                                                                          train_optimizer.param_groups[
                                                                                              0]['lr'],
                                                                                          total_loss / total_num))
    return total_loss / total_num

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

    model = CLIP().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=20,
                                                 num_training_steps=args.epochs)

    epoch_start = 0
    # training loop
    now_time = datetime.datetime.now()
    print(now_time)

    loss_values = []
    for epoch in range(epoch_start, args.epochs):
       train_loss = train(model, train_loader, optimizer, epoch, scheduler, args)
       loss_values.append(train_loss)

    with open('unsupervised_loss_values29-3.pkl', 'wb') as file:
       pickle.dump(loss_values, file)

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'unsupervised_'+args.type+'_model_last.pth')


    now_time = datetime.datetime.now()
    print(now_time)
    ## test
    test(model.net, test_loader, test_label)

    now_time = datetime.datetime.now()
    print(now_time)


