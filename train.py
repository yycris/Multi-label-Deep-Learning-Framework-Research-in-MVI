import torch
import os
import argparse
from dataloader import Dataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from model import CGAResNet
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import *
from sklearn.metrics import roc_auc_score

def data_list(datapath:str):
    train_images_path = []
    val_images_path = []
    test_images_path = []
    img_path = [datapath+'/train', datapath+'/valid', datapath+'/itest']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath+'/train':
                    train_images_path.append(os.path.join(root, file))
                elif path == datapath+'/valid':
                    val_images_path.append(os.path.join(root, file))
                else:
                    test_images_path.append(os.path.join(root, file))
    return train_images_path, val_images_path, test_images_path

def train_on_epochs(train_dataset:Dataset, val_dataset:Dataset, pre_train:str):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    net = CGAResNet.resnet(4, 4, mode='resnet18', pretrained=False)

    if pre_train != '':
        net.load_state_dict(torch.load(pre_train), strict=False)

    net.to(device)

    model_params = net.parameters()
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma, last_epoch=-1)

    # loss_function = nn.CrossEntropyLoss()

    classification_loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1, 0.1, 0.1]).to(device))
    regression_loss_fn = nn.MSELoss()

    writer = SummaryWriter('./logs/CGAResNet18')

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    train_loader = DataLoader(train_dataset, **config.dataset_params)
    val_loader = DataLoader(val_dataset, **config.dataset_params)

    train_steps = len(train_loader)
    val_steps = len(val_loader)

    for epoch in range(config.epoches):
        net.train()
        running_loss = 0.0
        test_loss = 0.0
        test_auc = 0.0
        train_bar = tqdm(train_loader)
        for step, (imgs_x, imgs_y, labs) in enumerate(train_bar):
            images_x = imgs_x.float()
            images_y = imgs_y.float()

            # labels = labs.to(torch.int64)
            labels = labs.to(torch.float32)

            optimizer.zero_grad()
            logits = net(images_x.to(device), images_y.to(device))

            # loss = loss_function(logits, labels.to(device))

            logits_classification = logits[:, :3]
            logits_regression = logits[:, 3]

            labels_classification = labels[:, :3]
            labels_regression = labels[:, 3]

            classification_loss = classification_loss_fn(logits_classification, labels_classification.to(device))
            regression_loss = regression_loss_fn(logits_regression, labels_regression.to(device))
            loss = classification_loss + (0.01 * regression_loss)


            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     config.epoches,
                                                                     loss)
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        net.eval()
        acc = 0.0
        test_label = []
        predict_pro = []
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for v_imgs_x, v_imgs_y, labs in val_bar:
                val_images_x = v_imgs_x.float()
                val_images_y = v_imgs_y.float()

                # val_labels = labs.to(torch.int64)
                val_labels = labs.to(torch.float32)

                # test_label.extend(val_labels.numpy())
                test_label.extend(val_labels[:, 0].numpy())

                outputs = net(val_images_x.to(device), val_images_y.to(device))

                # val_loss = loss_function(outputs, val_labels.to(device))

                val_logits_classification = outputs[:, :3]
                val_logits_regression = outputs[:, 3]

                val_labels_classification = val_labels[:, :3]
                val_labels_regression = val_labels[:, 3]

                val_classification_loss = classification_loss_fn(val_logits_classification, val_labels_classification.to(device))

                val_regression_loss = regression_loss_fn(val_logits_regression, val_labels_regression.to(device))

                val_loss = val_classification_loss + (0.01 * val_regression_loss)

                # predict_y = torch.max(outputs, dim=1)[1]

                predict_probability = torch.sigmoid(outputs[:, 0])
                predict_y = (predict_probability >= 0.5).long()

                # predict_probability = torch.softmax(outputs, dim=1)[:, 1]

                predict_pro.extend(predict_probability.cpu().numpy())

                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                acc += torch.eq(predict_y, val_labels[:, 0].to(device)).sum().item()

                test_loss += val_loss.item()
                val_bar.desc = "itest epoch[{}/{}]".format(epoch + 1,
                                                           config.epoches)
            val_accurate = acc / val_num
            test_auc = roc_auc_score(test_label, predict_pro)
            print('[epoch %d] train_loss: %.3f  val_loss: %.3f accurate: %.3f auc: %.3f' %
                  (epoch + 1, running_loss / train_steps, test_loss / val_steps, val_accurate, test_auc))
            ValLoss = test_loss / val_steps
            writer.add_scalars("Loss", {
                'Train': running_loss / train_steps,
                'Val': test_loss / val_steps
            }, epoch)
            writer.add_scalar("acc", val_accurate, epoch)
            writer.add_scalar("auc", test_auc, epoch)
            torch.save(net.state_dict(), config.save_path+'CGAResNet18 - '+str(epoch)+'.pth')

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./data')
    parser.add_argument('-v', '--vein_path', help='path to your vein', default='./data/vein')
    parser.add_argument('-l', '--label_path', help='path to your datasets label', default='./data/3y.csv')
    parser.add_argument('-r', '--pre_train', help='path to the pretrain weights', default='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    vein_path = args.vein_path
    label_path = args.label_path
    pre_train = args.pre_train

    train_transform = TrainTransforms()
    valid_transform = ValidTransforms()


    train_images_path, val_images_path, test_images_path = data_list(data_path)

    train_on_epochs(Dataset(train_images_path, vein_path, label_path,  transform=train_transform),
                    Dataset(val_images_path, vein_path, label_path,  transform=valid_transform),
                    pre_train)
