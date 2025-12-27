import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
import random

# ===================== 关键修改：设置缓存目录 =====================
# 将缓存目录改为当前目录的相对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(CURRENT_DIR, "cache")
# 创建缓存目录（如果不存在）
os.makedirs(CACHE_ROOT, exist_ok=True)
# 设置PyTorch预训练模型缓存目录
os.environ['TORCH_HOME'] = CACHE_ROOT  # 控制torchvision.models预训练模型的下载路径
# 可选：设置torch缓存目录（如需统一管理）
torch.hub.set_dir(os.path.join(CACHE_ROOT, "torch_hub"))
print(f"缓存目录设置为: {CACHE_ROOT}")


def get_datasets(root="ImageNet100"):
    label_names = sorted(os.listdir("ImageNet100/train"))
    img_paths_train = []
    labels_train = []
    # 注释掉测试集和验证集相关变量
    # img_paths_test, img_paths_val, labels_test, labels_val = [], [], [], []

    # 用于coreset采样的数据结构
    class_samples = {}

    # 仅处理train模式，注释掉val和test
    # for mode in ["train", "val", "test"]:
    for mode in ["train"]:
        dir_path1 = root + "/" + mode
        for dir_name2 in os.listdir(dir_path1):
            dir_path2 = dir_path1 + "/" + dir_name2
            class_id = label_names.index(dir_name2)

            # 收集该类所有样本路径
            class_img_paths = []
            for img_name in os.listdir(dir_path2):
                img_path = dir_path2 + "/" + img_name
                class_img_paths.append(img_path)

            # 对训练集进行coreset采样
            if mode == "train":
                if len(class_img_paths) > 400:
                    print(f"对类别 {dir_name2} 进行coreset采样 ({len(class_img_paths)} -> 400)")
                    sampled_paths = coreset_sampling_advanced(class_img_paths, 400)
                else:
                    sampled_paths = class_img_paths
                    print(f"类别 {dir_name2} 样本数不足400: {len(class_img_paths)} 个")

                # 将采样结果添加到训练集
                for img_path in sampled_paths:
                    img_paths_train.append(img_path)
                    labels_train.append(class_id)

                # 保存到class_samples供验证使用
                class_samples[class_id] = {
                    'original': len(class_img_paths),
                    'sampled': len(sampled_paths),
                    'name': dir_name2
                }
            # 注释掉验证集和测试集处理逻辑
            # else:
            #     # 验证集和测试集保持原样
            #     for img_path in class_img_paths:
            #         if mode == "test":
            #             img_paths_test.append(img_path)
            #             labels_test.append(class_id)
            #         else:
            #             img_paths_val.append(img_path)
            #             labels_val.append(class_id)

    # 打印采样统计信息
    print("\n" + "=" * 50)
    print("Coreset采样统计:")
    print("=" * 50)
    total_original = 0
    total_sampled = 0

    for class_id in sorted(class_samples.keys()):
        info = class_samples[class_id]
        total_original += info['original']
        total_sampled += info['sampled']
        reduction = ((info['original'] - info['sampled']) / info['original'] * 100) if info['original'] > 0 else 0
        print(
            f"类别 {info['name']} (ID: {class_id}): {info['original']} -> {info['sampled']} 个样本 (减少 {reduction:.1f}%)")

    print(f"\n总计: {total_original} -> {total_sampled} 个样本")
    print(f"减少比例: {(total_original - total_sampled) / total_original * 100:.1f}%")
    print(f"训练集总样本数: {len(img_paths_train)}")
    # 注释掉验证集和测试集统计
    # print(f"验证集总样本数: {len(img_paths_val)}")
    # print(f"测试集总样本数: {len(img_paths_test)}")
    print("=" * 50)

    # 创建datasets目录（如果不存在）
    datasets_dir = os.path.join(CURRENT_DIR, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # 保存到文件
    train_file = os.path.join(datasets_dir, "train.txt")
    # 注释掉测试集和验证集文件路径
    # test_file = os.path.join(datasets_dir, "test.txt")
    # val_file = os.path.join(datasets_dir, "val.txt")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for i in range(len(img_paths_train)):
            f.write(img_paths_train[i] + "\t" + str(labels_train[i]) + "\n")
    # 注释掉测试集和验证集文件写入
    # with open(test_file, "w", encoding="utf-8") as f:
    #     for i in range(len(img_paths_test)):
    #         f.write(img_paths_test[i] + "\t" + str(labels_test[i]) + "\n")
    # with open(val_file, "w", encoding="utf-8") as f:
    #     for i in range(len(img_paths_val)):
    #         f.write(img_paths_val[i] + "\t" + str(labels_val[i]) + "\n")

    print(f"\n数据集文件已保存到 {datasets_dir} 目录")
    print(f"仅生成了train.txt文件")
    
    # 仅返回train_file，注释掉其他返回值
    return train_file
    # return train_file, test_file, val_file


def coreset_sampling_advanced(image_paths, k, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    基于图像特征的coreset采样
    Args:
        image_paths: 图像路径列表
        k: 需要采样的数量
        batch_size: 批量处理大小
        device: 计算设备
    Returns:
        采样后的图像路径列表
    """
    if len(image_paths) <= k:
        return image_paths

    print(f"开始特征提取: {len(image_paths)} 张图片...")

    # 使用预训练的ResNet提取特征（模型将缓存到当前目录下的cache文件夹）
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的全连接层
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 批量提取特征
    features = []
    total_images = len(image_paths)

    with torch.no_grad():
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    feature = model(img_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    batch_features.append(feature)
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")
                    # 使用零向量作为占位符
                    batch_features.append(np.zeros(512))

            features.extend(batch_features)

            # 显示进度
            processed = min(i + batch_size, total_images)
            print(f"进度: {processed}/{total_images} ({processed / total_images * 100:.1f}%)", end='\r')

    print(f"\n特征提取完成，开始coreset采样...")

    features = np.array(features)

    # 执行coreset采样 (k-medoids风格)
    selected_indices = []

    # 1. 随机选择第一个中心点
    selected_indices.append(random.randint(0, len(features) - 1))

    # 2. 迭代选择剩余的点
    iteration = 1
    while len(selected_indices) < k:
        print(f"采样进度: {len(selected_indices)}/{k} (迭代 {iteration})", end='\r')

        # 计算所有点到已选点的最小距离
        min_distances = np.full(len(features), np.inf)

        # 优化：批量计算距离
        for selected_idx in selected_indices:
            # 计算所有点到当前选中点的距离
            distances = np.linalg.norm(features - features[selected_idx], axis=1)
            min_distances = np.minimum(min_distances, distances)

        # 将已选点的距离设为负无穷，避免重复选择
        min_distances[selected_indices] = -np.inf

        # 选择具有最大最小距离的点
        next_point = np.argmax(min_distances)
        selected_indices.append(next_point)
        iteration += 1

    print(f"\nCoreset采样完成!")

    # 返回采样结果
    return [image_paths[i] for i in selected_indices]


def simple_coreset_sampling(image_paths, k):
    """
    简化的coreset采样（用于测试或快速运行）
    Args:
        image_paths: 图像路径列表
        k: 需要采样的数量
    Returns:
        采样后的图像路径列表
    """
    if len(image_paths) <= k:
        return image_paths

    print(f"使用简化coreset采样: {len(image_paths)} -> {k}")

    # 使用文件名作为简单特征
    features = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # 将文件名转换为数值特征
        feature = [ord(c) for c in filename[:50]]  # 取前50个字符
        if len(feature) < 50:
            feature += [0] * (50 - len(feature))  # 填充到50维
        features.append(feature[:50])  # 确保正好50维

    features = np.array(features)

    # 执行coreset采样
    selected_indices = []
    selected_indices.append(random.randint(0, len(features) - 1))

    while len(selected_indices) < k:
        min_distances = np.full(len(features), np.inf)

        for selected_idx in selected_indices:
            distances = np.linalg.norm(features - features[selected_idx], axis=1)
            min_distances = np.minimum(min_distances, distances)

        min_distances[selected_indices] = -np.inf
        next_point = np.argmax(min_distances)
        selected_indices.append(next_point)

    return [image_paths[i] for i in selected_indices]


class DataGenerator(Dataset):
    def __init__(self, root, mode=None):
        super(DataGenerator, self).__init__()
        self.root = root
        
        # 自动推断模式：如果未指定mode，尝试从文件名中推断
        if mode is None:
            if "train" in root:
                self.mode = "train"
            elif "val" in root or "test" in root:
                self.mode = "val"
            else:
                self.mode = "train" # 默认为训练模式
        else:
            self.mode = mode

        print(f"[DataGenerator] 加载文件: {root} | 模式: {self.mode}")

        # ====================================================================
        # 核心修改：区分训练和验证的数据增强策略
        # ====================================================================
        if self.mode == "train":
            # 训练集：强数据增强，防止过拟合
            self.transforms = transforms.Compose([
                # 1. 随机裁剪并缩放：迫使模型看物体的不同局部，学习尺度不变性
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), 
                
                # 2. 随机水平翻转：学习左右对称性
                transforms.RandomHorizontalFlip(),
                
                # 3. 颜色抖动：改变亮度/对比度/饱和度，防止模型死记颜色
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                
                # 4. 转为Tensor并归一化
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 验证/测试集：确定性处理，确保评估公平稳定
            self.transforms = transforms.Compose([
                # 1. 先缩放到256 (ImageNet标准做法)
                transforms.Resize(256),
                
                # 2. 中心裁剪出224
                transforms.CenterCrop(224),
                
                # 3. 转为Tensor并归一化
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.img_paths, self.labels = self.get_datasets()

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        # 确保图片转为RGB（处理灰度图或RGBA图）
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            label = torch.tensor(self.labels[item], dtype=torch.long)
            return img, label
        except Exception as e:
            # 容错处理：如果某张图坏了，打印错误并随机返回下一张
            print(f"读取图片出错: {img_path}, error: {e}")
            return self.__getitem__((item + 1) % len(self))

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        img_paths, labels = [], []
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"数据集文件未找到: {self.root}")
            
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                line_split = line.split("\t")
                img_paths.append(line_split[0])
                labels.append(int(line_split[-1]))

        return img_paths, labels


if __name__ == '__main__':
    # 测试采样功能
    # 仅接收train_file，注释掉其他返回值
    train_file = get_datasets()
    # train_file, test_file, val_file = get_datasets()

    # 可选：验证采样结果
    print("\n验证采样结果...")
    train_dataset = DataGenerator(train_file)
    print(f"训练集加载成功: {len(train_dataset)} 个样本")

    # 统计每个类别的样本数
    class_counts = {}
    for _, label in train_dataset:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"\n训练集中各类别样本数:")
    for class_id in sorted(class_counts.keys()):
        print(f"类别 {class_id}: {class_counts[class_id]} 个样本")
    
    print(f"\n缓存目录: {CACHE_ROOT}")
    print(f"数据集文件:")
    print(f"  - 训练集: {train_file}")
    # 注释掉测试集和验证集打印
    # print(f"  - 测试集: {test_file}")
    # print(f"  - 验证集: {val_file}")