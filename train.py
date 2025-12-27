import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import DataGenerator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from nets.swin_transformer import SwinTransformer
from nets.lyj_swin_transformer import LYJSwinTransformer
import warnings
import os
import glob
warnings.filterwarnings("ignore")


class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=10, min_delta=0.002, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_acc, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.save_best_model(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_model(model)
            self.counter = 0
            
        return self.early_stop
    
    def save_best_model(self, model):
        if self.restore_best_weights:
            self.best_model_state = model.state_dict().copy()
    
    def restore_model(self, model):
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载checkpoint以恢复训练"""
    if not os.path.exists(checkpoint_path):
        print(f"警告: checkpoint文件不存在: {checkpoint_path}")
        return None
    
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 恢复模型
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 兼容旧的只保存权重的格式
        model.load_state_dict(checkpoint)
    
    # 恢复优化器和学习率调度器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 提取训练信息
    train_info = {}
    if 'epoch' in checkpoint:
        train_info['epoch'] = checkpoint['epoch']
    
    if 'train_accs' in checkpoint:
        train_info['train_accs'] = checkpoint['train_accs']
    if 'train_losses' in checkpoint:
        train_info['train_losses'] = checkpoint['train_losses']
    if 'val_accs' in checkpoint:
        train_info['val_accs'] = checkpoint['val_accs']
    if 'val_losses' in checkpoint:
        train_info['val_losses'] = checkpoint['val_losses']
    if 'max_val_acc' in checkpoint:
        train_info['max_val_acc'] = checkpoint['max_val_acc']
    if 'best_epoch' in checkpoint:
        train_info['best_epoch'] = checkpoint['best_epoch']
    if 'no_improve_count' in checkpoint:
        train_info['no_improve_count'] = checkpoint['no_improve_count']
    
    print(f"从epoch {checkpoint.get('epoch', 'N/A')}恢复训练")
    return train_info


def save_checkpoint(model, optimizer, scheduler, epoch, model_name, 
                    train_accs, train_losses, val_accs, val_losses,
                    max_val_acc, best_epoch, no_improve_count, 
                    save_dir="models", max_checkpoints=5):
    """保存完整的训练checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch + 1,  # 保存下一个要训练的epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_accs': train_accs,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'val_losses': val_losses,
        'max_val_acc': max_val_acc,
        'best_epoch': best_epoch,
        'no_improve_count': no_improve_count
    }
    
    # 保存当前checkpoint
    checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch{epoch+1:03d}_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint已保存: {checkpoint_path}")
    
    # 清理旧的checkpoint，只保留最新的max_checkpoints个
    cleanup_old_checkpoints(model_name, save_dir, max_checkpoints)
    
    return checkpoint_path


def cleanup_old_checkpoints(model_name, save_dir="models", max_checkpoints=5):
    """清理旧的checkpoint，只保留最新的几个"""
    checkpoint_pattern = os.path.join(save_dir, f"{model_name}_epoch*_checkpoint.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) > max_checkpoints:
        # 按epoch数排序
        def get_epoch_num(path):
            filename = os.path.basename(path)
            # 提取epoch数
            epoch_str = filename.split('_')[1].replace('epoch', '').replace('checkpoint.pth', '').strip('_')
            return int(epoch_str)
        
        try:
            checkpoints_sorted = sorted(checkpoints, key=get_epoch_num)
            # 删除旧的checkpoint
            for old_checkpoint in checkpoints_sorted[:-max_checkpoints]:
                os.remove(old_checkpoint)
                print(f"  清理旧checkpoint: {os.path.basename(old_checkpoint)}")
        except (ValueError, IndexError):
            # 如果文件名解析失败，按修改时间排序
            checkpoints_sorted = sorted(checkpoints, key=os.path.getmtime)
            for old_checkpoint in checkpoints_sorted[:-max_checkpoints]:
                os.remove(old_checkpoint)
                print(f"  清理旧checkpoint: {os.path.basename(old_checkpoint)}")


def find_latest_checkpoint(model_name, save_dir="models"):
    """查找最新的checkpoint文件"""
    checkpoint_pattern = os.path.join(save_dir, f"{model_name}_epoch*_checkpoint.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # 按epoch数排序
    def get_epoch_num(path):
        filename = os.path.basename(path)
        # 提取epoch数
        epoch_str = filename.split('_')[1].replace('epoch', '').replace('checkpoint.pth', '').strip('_')
        return int(epoch_str) if epoch_str.isdigit() else 0
    
    try:
        latest_checkpoint = max(checkpoints, key=get_epoch_num)
        return latest_checkpoint
    except (ValueError, IndexError):
        # 如果文件名解析失败，按修改时间排序
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        return latest_checkpoint


def train(model_name, resume_from=None, total_epochs=50, checkpoint_interval=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # 初始化模型 - 保持文档1的模型初始化方式
    if model_name == "swin-t":
        model = SwinTransformer(num_classes=100).to(device)
    elif model_name == "lyj_swin-t":
        model = LYJSwinTransformer(num_classes=100).to(device)
    else:
        raise ValueError("model name must be swin-t or lyj_swin-t!")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 数据加载器 
    train_loader = DataLoader(
        DataGenerator(root="datasets/train.txt"), 
        batch_size=32,  
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        DataGenerator(root="datasets/val.txt"), 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 优化器设置 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # 使用余弦退火学习率调度 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )
    
    # 使用标签平滑防止过拟合  
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 初始化训练变量
    start_epoch = 0
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    max_val_acc = 0
    best_epoch = 0
    no_improve_count = 0
    
    # 尝试从checkpoint恢复
    if resume_from is not None:
        train_info = load_checkpoint(resume_from, model, optimizer, scheduler)
        if train_info:
            start_epoch = train_info.get('epoch', 0)
            train_accs = train_info.get('train_accs', [])
            train_losses = train_info.get('train_losses', [])
            val_accs = train_info.get('val_accs', [])
            val_losses = train_info.get('val_losses', [])
            max_val_acc = train_info.get('max_val_acc', 0)
            best_epoch = train_info.get('best_epoch', 0)
            no_improve_count = train_info.get('no_improve_count', 0)
            print(f"成功从checkpoint恢复训练状态")
            print(f"将从epoch {start_epoch}继续训练")
    else:
        # 自动查找最新的checkpoint
        latest_checkpoint = find_latest_checkpoint(model_name)
        if latest_checkpoint:
            response = input(f"找到最新checkpoint: {latest_checkpoint}\n是否从该checkpoint恢复训练? (y/n): ")
            if response.lower() in ['y', 'yes', '是']:
                train_info = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
                if train_info:
                    start_epoch = train_info.get('epoch', 0)
                    train_accs = train_info.get('train_accs', [])
                    train_losses = train_info.get('train_losses', [])
                    val_accs = train_info.get('val_accs', [])
                    val_losses = train_info.get('val_losses', [])
                    max_val_acc = train_info.get('max_val_acc', 0)
                    best_epoch = train_info.get('best_epoch', 0)
                    no_improve_count = train_info.get('no_improve_count', 0)
                    print(f"成功从checkpoint恢复训练状态")
                    print(f"将从epoch {start_epoch}继续训练")
    
    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    print(f"开始训练模型: {model_name}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"训练参数: lr=1e-3, batch_size=32, epochs={total_epochs}")
    print(f"当前起始轮次: {start_epoch}")
    print(f"Checkpoint保存间隔: 每{checkpoint_interval}个epoch")
    
    for epoch in range(start_epoch, total_epochs):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_func, device, epoch
        )
        val_acc, val_loss = get_val_result(model, val_loader, loss_func, device)
        
        # 更新学习率
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 更新最佳模型
        if val_acc > max_val_acc:
            # 保存最佳模型权重
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, f"models/{model_name}_best.pth")
            
            # 保存完整的最佳模型checkpoint
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accs': train_accs,
                'train_losses': train_losses,
                'val_accs': val_accs,
                'val_losses': val_losses,
                'max_val_acc': val_acc,
                'best_epoch': epoch + 1,
                'no_improve_count': 0
            }
            torch.save(best_checkpoint, f"models/{model_name}_best_checkpoint.pth")
            
            max_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve_count = 0
            print(f" 保存最佳模型 (Acc: {val_acc:.4f})")
        else:
            no_improve_count += 1
        
        # 每checkpoint_interval个epoch保存一次完整的checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == total_epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                model_name=model_name,
                train_accs=train_accs,
                train_losses=train_losses,
                val_accs=val_accs,
                val_losses=val_losses,
                max_val_acc=max_val_acc,
                best_epoch=best_epoch,
                no_improve_count=no_improve_count
            )
        
        # 检查早停
        if early_stopping(val_acc, model):
            print(f"\n早停触发! 在epoch {epoch+1}停止训练")
            print(f"最佳验证准确率: {early_stopping.best_score:.4f}")
            
            # 恢复最佳模型权重
            if early_stopping.restore_best_weights:
                early_stopping.restore_model(model)
                print("已恢复最佳模型权重")
            
            # 保存最终模型
            torch.save(model.state_dict(), f"models/{model_name}_final.pth")
            # 保存最终的checkpoint
            final_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accs': train_accs,
                'train_losses': train_losses,
                'val_accs': val_accs,
                'val_losses': val_losses,
                'max_val_acc': max_val_acc,
                'best_epoch': best_epoch,
                'no_improve_count': no_improve_count,
                'early_stopped': True
            }
            torch.save(final_checkpoint, f"models/{model_name}_final_checkpoint.pth")
            
            break
    
    # 如果正常完成所有epoch
    if not early_stopping.early_stop:
        # 恢复最佳模型并保存最终版本
        if early_stopping.restore_best_weights and early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
            print("已恢复最佳模型权重")
        
        torch.save(model.state_dict(), f"models/{model_name}_final.pth")
        # 保存最终的checkpoint
        final_checkpoint = {
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_accs': train_accs,
            'train_losses': train_losses,
            'val_accs': val_accs,
            'val_losses': val_losses,
            'max_val_acc': max_val_acc,
            'best_epoch': best_epoch,
            'no_improve_count': no_improve_count,
            'early_stopped': False
        }
        torch.save(final_checkpoint, f"models/{model_name}_final_checkpoint.pth")
        
        print(f"\n训练完成! 最佳验证准确率: {max_val_acc:.4f}")
    
    # 绘制结果
    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name)
    
    # 保存训练历史
    save_results(train_accs, train_losses, val_accs, val_losses, model_name)
    
    return model


def train_one_epoch(model, train_loader, optimizer, loss_func, device, epoch):
    model.train()
    
    data = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch, (x, y) in enumerate(data):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        prob = model(x)
        loss = loss_func(prob, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            pred = torch.argmax(prob, dim=1)
            batch_correct = (pred == y).sum().item()
            
            total_loss += loss.item() * x.size(0)
            total_correct += batch_correct
            total_samples += x.size(0)
            
            # 更新进度条
            if batch % 20 == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                avg_acc = total_correct / total_samples if total_samples > 0 else 0
                data.set_postfix_str(f"loss:{avg_loss:.4f}, acc:{avg_acc:.3f}")
    
    # 计算epoch指标
    epoch_loss = total_loss / total_samples if total_samples > 0 else 0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return epoch_acc, epoch_loss


def get_val_result(model, val_loader, loss_func, device):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            prob = model(x)
            loss = loss_func(prob, y)
            
            pred = torch.argmax(prob, dim=1)
            batch_correct = (pred == y).sum().item()
            
            total_loss += loss.item() * x.size(0)
            total_correct += batch_correct
            total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_acc, avg_loss


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name):
    plt.figure(figsize=(12, 5))
    
    epochs = range(1, len(train_accs) + 1)
    
    # 准确率图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, "r-", label="Train", linewidth=2)
    plt.plot(epochs, val_accs, "g-", label="Val", linewidth=2)
    
    best_epoch = val_accs.index(max(val_accs)) + 1
    plt.scatter(best_epoch, max(val_accs), color='blue', s=100, zorder=5, 
                label=f'Best: {max(val_accs):.4f}')
    
    plt.title(f"{model_name} - Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 损失图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, "r-", label="Train", linewidth=2)
    plt.plot(epochs, val_losses, "g-", label="Val", linewidth=2)
    
    plt.title(f"{model_name} - Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_results.jpg", dpi=150, bbox_inches='tight')
    plt.close()


def save_results(train_accs, train_losses, val_accs, val_losses, model_name):
    df = pd.DataFrame({
        "epoch": range(1, len(train_accs) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    })
    df.to_excel(f"results/{model_name}_results.xlsx", index=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print(f"训练摘要 - {model_name}")
    print("="*60)
    print(f"训练轮次: {len(train_accs)}")
    print(f"最终训练准确率: {train_accs[-1]:.4f}")
    print(f"最终验证准确率: {val_accs[-1]:.4f}")
    print(f"最佳验证准确率: {max(val_accs):.4f} (Epoch {val_accs.index(max(val_accs)) + 1})")
    
    # 计算提升
    if len(train_accs) > 1:
        acc_gain = val_accs[-1] - val_accs[0]
        print(f"验证准确率提升: {acc_gain:.4f} ({acc_gain*100:.2f}%)")
    print("="*60)


def resume_training(model_name, checkpoint_path=None, total_epochs=50):
    """恢复训练的主函数"""
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(model_name)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n从checkpoint恢复训练")
        print(f"Checkpoint文件: {checkpoint_path}")
        
        # 获取checkpoint的epoch信息
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)
        completed_epochs = start_epoch - 1
        
        if completed_epochs >= total_epochs:
            print(f"警告: checkpoint已经训练了{completed_epochs}个epoch，超过了总epochs({total_epochs})")
            response = input("是否继续训练? (y/n): ")
            if response.lower() not in ['y', 'yes', '是']:
                return
        
        # 继续训练
        train(model_name, resume_from=checkpoint_path, total_epochs=total_epochs)
    else:
        print(f"没有找到可用的checkpoint，开始新训练")
        train(model_name, total_epochs=total_epochs)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 训练参数
    TOTAL_EPOCHS = 50
    CHECKPOINT_INTERVAL = 10
    
    # 检查是否要恢复训练
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--resume":
            checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
            resume_training("lyj_swin-t", checkpoint_path, TOTAL_EPOCHS)
        elif sys.argv[1] == "--train":
            train("lyj_swin-t", total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
        else:
            print("使用方法:")
            print("  python train.py --train                 # 开始新训练")
            print("  python train.py --resume [checkpoint]   # 从checkpoint恢复训练")
            print("  python train.py                         # 交互式选择")
    else:
        # 交互式选择
        print("请选择训练模式:")
        print("1. 开始新训练")
        print("2. 从最近的checkpoint恢复训练")
        print("3. 指定checkpoint文件恢复训练")
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            # 训练模型 - 根据你的需求选择
            train("lyj_swin-t", total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
            train("swin-t", total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
        elif choice == "2":
            resume_training("lyj_swin-t", total_epochs=TOTAL_EPOCHS)
            resume_training("swin-t", total_epochs=TOTAL_EPOCHS)
        elif choice == "3":
            checkpoint_path = input("请输入checkpoint文件路径: ").strip()
            resume_training("lyj_swin-t", checkpoint_path, TOTAL_EPOCHS)
            resume_training("swin-t", checkpoint_path, TOTAL_EPOCHS)
        else:
            print("无效选项，开始新训练...")
            train("lyj_swin-t", total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
            train("swin-t", total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)