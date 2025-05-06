import torch
import os
import numpy as np
from PIL import Image
from model import CGAResNet
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
import pandas as pd
import torchvision.transforms as t
from sklearn.utils import resample
from utils import *

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    train_root = './data/train'
    itest_root = './data/itest'
    val_root = './data/itest'
    vein_root = './data/vein'
    fat_root = './data/fat'
    etest_root = './data/etest'
    etest_vein_root = './data/etest_vein'
    etest_fat_root = './data/etest_fat'

    train = []
    itest = []
    val = []
    etest = []

    # label_root = './data/3y.csv'
    label_root = './data/2y.csv'
    label_data = pd.read_csv(os.path.join(label_root))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = CGAResNet.resnet(in_channels=4, num_classes=4, mode='resnet18', pretrained=False).to(device)

    for root, dirs, files in os.walk(train_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                train.append(os.path.join(root, file))

    for root, dirs, files in os.walk(itest_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                itest.append(os.path.join(root, file))

    for root, dirs, files in os.walk(val_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                val.append(os.path.join(root, file))

    for root, dirs, files in os.walk(etest_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                etest.append(os.path.join(root, file))

    a = etest
    maxAuc = 0.0
    index_final = 0

    predict_pro = []
    predict_cla = []
    test_label = []

    model_weight_path = f"./result/CGAResNet18/cgaresnet18_tumor_fat_LOA_multiLabel.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()
    with torch.no_grad():
        for img_path in a:
            img = Image.open(img_path).convert('RGB')
            i = img_path[13:]
            i = i[:-4]
            img_v = Image.open(etest_vein_root + '/' + i + '.png').convert('RGB')

            resize_transform = T.Resize([224, 224])
            img, img_v = resize_transform(img, img_v)

            img_fat = Image.open(etest_fat_root + '/' + i + '.png').convert('L')
            resize_transform_1 = t.Resize((224, 224))
            img_fat = resize_transform_1(img_fat)
            img_np = np.array(img)  # (H, W, 3)
            img_v_np = np.array(img_v)  # (H, W, 3)
            img_fat_np = np.array(img_fat)  # (H, W)
            img_fat_np = np.expand_dims(img_fat_np, axis=2)  # (H, W, 1)
            img_x_fat_np = np.concatenate((img_np, img_fat_np), axis=2)  # (H, W, 4)
            img_v_fat_np = np.concatenate((img_v_np, img_fat_np), axis=2)  # (H, W, 4)
            img = Image.fromarray(img_x_fat_np)
            img_v = Image.fromarray(img_v_fat_np)

            train_transform_1 = t.Compose([
                t.ToTensor(),
                # t.Normalize(mean=[0.04488417, 0.04488417, 0.01361942],
                #             std=[0.14777726, 0.14777726, 0.11397097]),
                t.Normalize(mean=[0.04488417, 0.04488417, 0.01361942, 0.00050994], std=[0.14777726, 0.14777726, 0.11397097, 0.00750913])
            ])
            train_transform_2 = t.Compose([
                t.ToTensor(),
                # t.Normalize(mean=[0.05089118, 0.05089118, 0.013708],
                #             std=[0.16724654, 0.16724654, 0.11444444]),

                t.Normalize(mean=[0.05089118, 0.05089118, 0.013708, 0.00050994],
                            std=[0.16724654, 0.16724654, 0.11444444, 0.00750913])
            ])
            img = train_transform_1(img)
            img_v = train_transform_2(img_v)
            img = torch.unsqueeze(img, dim=0).float()
            img_v = torch.unsqueeze(img_v, dim=0).float()


            # net.flatten.register_forward_hook(get_activation('flatten'))
            output = torch.squeeze(net(img.to(device), img_v.to(device)))
            label = label_data.loc[label_data['病历号'] == int(i), ['group1']].values.item()

            # predict = torch.softmax(output, dim=0)
            # output = predict[1]
            # predict = predict[1]

            predict = torch.sigmoid(output[0])
            output = predict

            threshold = 0.5
            output = torch.where(output >= threshold, torch.ones_like(output), output)
            output = torch.where(output < threshold, torch.zeros_like(output), output)
            predict_class = output.cpu().numpy()
            predict_probability = predict.cpu().numpy()
            predict_pro.append(predict_probability.item())
            predict_cla.append(predict_class.item())
            test_label.append(label)

            # list = []
            # list.append(int(i))
            # list.append(label)
            # tensor = activation['flatten'].cpu()
            # tensor = torch.squeeze(tensor)
            # for x in range(0, len(tensor)):
            #     list.append(tensor[x].float().item())
            # with open("./result/visualized/t-sne/train_tsne.csv", "a", newline='', encoding='utf-8') as file:
            #     writer = csv.writer(file, delimiter=',')
            #     writer.writerow(list)

    matrix = confusion_matrix(test_label, predict_cla, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = accuracy_score(test_label, predict_cla)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    AUC = roc_auc_score(test_label, predict_pro)

    y_true = np.array(test_label)
    y_pred = np.array(predict_cla)
    y_score = np.array(predict_pro)

    n_bootstraps = 1000
    auc_scores = []
    acc_scores = []
    sen_scores = []
    spe_scores = []
    ppv_scores = []
    npv_scores = []

    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)))
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]
        y_score_sampled = y_score[indices]

        auc_sampled = roc_auc_score(y_true_sampled, y_score_sampled)
        acc_sampled = accuracy_score(y_true_sampled, y_pred_sampled)
        sen_sampled = recall_score(y_true_sampled, y_pred_sampled)
        conf_matrix_sampled = confusion_matrix(y_true_sampled, y_pred_sampled)
        tn_sampled, fp_sampled, fn_sampled, tp_sampled = conf_matrix_sampled.ravel()
        spe_sampled = tn_sampled / (tn_sampled + fp_sampled)
        ppv_sampled = tp_sampled / (tp_sampled + fp_sampled)
        npv_sampled = tn_sampled / (tn_sampled + fn_sampled)

        auc_scores.append(auc_sampled)
        acc_scores.append(acc_sampled)
        sen_scores.append(sen_sampled)
        spe_scores.append(spe_sampled)
        ppv_scores.append(ppv_sampled)
        npv_scores.append(npv_sampled)

    alpha = 0.95
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100

    auc_lower_bound = max(0.0, np.percentile(auc_scores, lower_percentile))
    auc_upper_bound = min(1.0, np.percentile(auc_scores, upper_percentile))

    acc_lower_bound = max(0.0, np.percentile(acc_scores, lower_percentile))
    acc_upper_bound = min(1.0, np.percentile(acc_scores, upper_percentile))

    sen_lower_bound = max(0.0, np.percentile(sen_scores, lower_percentile))
    sen_upper_bound = min(1.0, np.percentile(sen_scores, upper_percentile))

    spe_lower_bound = max(0.0, np.percentile(spe_scores, lower_percentile))
    spe_upper_bound = min(1.0, np.percentile(spe_scores, upper_percentile))

    ppv_lower_bound = max(0.0, np.percentile(ppv_scores, lower_percentile))
    ppv_upper_bound = min(1.0, np.percentile(ppv_scores, upper_percentile))

    npv_lower_bound = max(0.0, np.percentile(npv_scores, lower_percentile))
    npv_upper_bound = min(1.0, np.percentile(npv_scores, upper_percentile))


    print('AUC:', AUC)
    print(f"95% 置信区间 (AUC): ({auc_lower_bound:.4f}, {auc_upper_bound:.4f})")
    print('acc: ', acc)
    print(f"95% 置信区间 (Accuracy): ({acc_lower_bound:.4f}, {acc_upper_bound:.4f})")
    print('sensitivity: ', sen)
    print(f"95% 置信区间 (Sensitivity): ({sen_lower_bound:.4f}, {sen_upper_bound:.4f})")
    print('specificity: ', spe)
    print(f"95% 置信区间 (Specificity): ({spe_lower_bound:.4f}, {spe_upper_bound:.4f})")
    print('ppv:', ppv)
    print(f"95% 置信区间 (PPV): ({ppv_lower_bound:.4f}, {ppv_upper_bound:.4f})")
    print('npv:', npv)
    print(f"95% 置信区间 (NPV): ({npv_lower_bound:.4f}, {npv_upper_bound:.4f})")

if __name__ == '__main__':
    main()