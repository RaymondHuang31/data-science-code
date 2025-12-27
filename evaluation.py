import os
import torch
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.swin_transformer import SwinTransformer
from datasets import DataGenerator
from torch.utils.data import DataLoader
from nets.lyj_swin_transformer import LYJSwinTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, confusion_matrix, \
    precision_recall_curve, classification_report

rc = {'font.sans-serif': 'SimHei', 'axes.unicode_minus': False}
seaborn.set(context='notebook', style='ticks', rc=rc)


def one_hot(y, num_classes=100):
    y_ = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_[i, int(y[i])] = 1

    return y_


def softmax(y):
    y_exp = np.exp(y)
    for i in range(len(y)):
        y_exp[i, :] = y_exp[i, :] / np.sum(y_exp[i, :])

    return y_exp


def get_test_result(model_path, model_name, columns, root="datasets/test.txt"):
    print(model_name)
    device = torch.device("cuda")
    if model_name == "swin-t":
        model = SwinTransformer(num_classes=100).to(device)
    elif model_name == "lyj_swin-t":
        model = LYJSwinTransformer(num_classes=100).to(device)
    else:
        raise ValueError("model name must be swin-t!")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(DataGenerator(root=root), batch_size=32, shuffle=False)
    data = tqdm(test_loader)
    labels_true, labels_pred, labels_prob = np.array([]), np.array([]), []
    with torch.no_grad():
        for x, y in data:
            datasets_test = x.to(device)
            prob = model(datasets_test)
            labels_prob.append(prob.cpu().numpy())
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)

    labels_prob = softmax(np.concatenate(labels_prob, axis=0))
    labels_onehot = one_hot(labels_true, num_classes=100)

    accuracy = accuracy_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred, average="macro")
    recall = recall_score(labels_true, labels_pred, average="macro")
    f1 = f1_score(labels_true, labels_pred, average="macro")
    print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1:{f1}")
    # print(classification_report(labels_true, labels_pred, target_names=label_names))
    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot([0, 1], [0, 1], "r--")
    fpr, tpr, _ = roc_curve(labels_onehot.ravel(), labels_prob.ravel())
    plt.plot(fpr, tpr, "g", label=f"AUC:{auc(fpr, tpr):.3f}", linewidth=5.0)
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    plt.xlabel("FPR", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=22)
    plt.ylabel("TPR", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=25)
    plt.xticks(weight='bold', fontproperties='Times New Roman')
    plt.yticks(weight='bold', fontproperties='Times New Roman')
    plt.tick_params(pad=1.5)
    plt.tick_params(labelsize=20, width=1)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_roc_curve.jpg", dpi=300)

    plt.figure(figsize=(10, 10), dpi=300)
    p, r, _ = precision_recall_curve(labels_onehot.ravel(), labels_prob.ravel())
    plt.plot(p, r, "g", linewidth=5.0)
    plt.xlabel("Precision", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=22)
    plt.ylabel("Recall", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=25)
    plt.xticks(weight='bold', fontproperties='Times New Roman')
    plt.yticks(weight='bold', fontproperties='Times New Roman')
    plt.tick_params(pad=1.5)
    plt.tick_params(labelsize=20, width=1)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_pr_curve.jpg", dpi=300)

    matrix = pd.DataFrame(confusion_matrix(labels_true, labels_pred, normalize="true"), columns=columns, index=columns)
    plt.figure(figsize=(15, 15), dpi=300)
    seaborn.heatmap(matrix, annot=False, cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig(f"images/{model_name}_confusion_matrix.jpg", dpi=300)


if __name__ == '__main__':
    label_names = sorted(os.listdir("ImageNet100/train"))
    get_test_result(model_path=f"models/swin-t_best.pth", model_name="swin-t", columns=label_names)
    get_test_result(model_path=f"models/lyj_swin-t_best.pth", model_name="lyj_swin-t", columns=label_names)


