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

# 可选：启用混合精度训练（如果GPU支持）
# USE_AMP = True
# if USE_AMP:
#     from torch.cuda.amp import autocast, GradScaler


class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience: 容忍的验证集性能不提升的epoch数
            min_delta: 被认为是提升的最小变化量
            restore_best_weights: 是否在早停时恢复最佳权重
        """
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


def set_strategy(model, model_name):

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True

    if model_name == "swin-t":

        for param in model.layers[-1].parameters():
            param.requires_grad = True


    elif model_name == "lyj_swin-t":
        
        for param in model.patch_embed.proj[1].parameters():  
            param.requires_grad = True

        if hasattr(model, 'sspp'):
            for param in model.sspp.parameters():
                param.requires_grad = True

        for param in model.layers[-1].parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

def train(model_name, resume_from=None, total_epochs=20, checkpoint_interval=5):
    device = torch.device("cuda")
    
    # 初始化混合精度训练
    # scaler = GradScaler() if USE_AMP else None
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    if model_name == "swin-t":
        model = SwinTransformer(num_classes=100).to(device)
    elif model_name == "lyj_swin-t":
        model = LYJSwinTransformer(num_classes=100).to(device)
    else:
        raise ValueError("model name must be swin-t or lyj_swin-t!")
    
    state_dict = torch.load("models/swin_pretrained.pth")
    del state_dict["head.bias"]
    del state_dict["head.weight"]
    model.load_state_dict(state_dict, strict=False)

    set_strategy(model, model_name)


    train_loader = DataLoader(
        DataGenerator(root="datasets/train.txt"), 
        batch_size=32, 
        shuffle=True, 
        num_workers=8,      
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context='spawn'
    )
    
    val_loader = DataLoader(
        DataGenerator(root="datasets/val.txt"), 
        batch_size=32, 
        shuffle=False,
        num_workers=4,  
        pin_memory=True
    )
    
    optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-4,
    weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_func = nn.CrossEntropyLoss()
    
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
            print("使用预训练权重开始新训练")
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
            else:
                print("开始新训练")
    
    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name}")
    print(f"总训练轮次: {total_epochs}")
    print(f"当前起始轮次: {start_epoch}")
    print(f"Checkpoint保存间隔: 每{checkpoint_interval}个epoch")
    print(f"早停机制: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{total_epochs}")
        print(f"{'='*60}")
        
        # 训练一个epoch
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            loss_func, device, epoch
        )
        
        # 验证
        val_acc, val_loss = get_val_result(model, val_loader, loss_func, device)
        
        # 保存训练结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印当前epoch结果
        print(f"\n[Epoch {epoch+1}] train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}, "
              f"val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        
        # 更新最佳模型
        if val_acc > max_val_acc:
            # 保存最佳模型权重
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
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
            print(f"  ✓ 新的最佳模型已保存，验证准确率: {val_acc:.4f}")
        else:
            no_improve_count += 1
            print(f"  - 验证集准确率 {no_improve_count} 个 Epoch 未提升")
        
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
        
        # 检查是否需要早停
        if early_stopping(val_acc, model):
            print(f"\n{'='*60}")
            print(f"早停触发! 在epoch {epoch+1}停止训练")
            print(f"最佳验证准确率: {early_stopping.best_score:.4f} 出现在第 {best_epoch} 轮")
            
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
            
            # 绘制图表
            plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name, early_stopped=True)
            
            # 保存训练历史
            save_training_history(train_accs, train_losses, val_accs, val_losses, model_name)
            
            # 打印训练摘要
            print_training_summary(model_name, len(train_accs), max_val_acc, best_epoch, 
                                 train_accs[-1], val_accs[-1], early_stopped=True)
            break
    
    # 如果正常完成所有epoch
    if not early_stopping.early_stop:
        # 保存最终模型
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
        
        print(f"\n{'='*60}")
        print(f"训练完成! 最佳验证准确率: {max_val_acc:.4f} 出现在第 {best_epoch} 轮")
        
        # 绘制图表
        plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name, early_stopped=False)
        
        # 保存训练历史
        save_training_history(train_accs, train_losses, val_accs, val_losses, model_name)
        
        # 打印训练摘要
        print_training_summary(model_name, len(train_accs), max_val_acc, best_epoch, 
                             train_accs[-1], val_accs[-1], early_stopped=False)
    
    return model


def train_one_epoch(model, train_loader, optimizer, scheduler, 
                   loss_func, device, epoch):
    model.train()
    
    # 减少tqdm刷新频率，每0.3秒刷新一次
    data = tqdm(train_loader, mininterval=0.3, desc=f"Epoch {epoch+1}")
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch, (x, y) in enumerate(data):
        # 使用non_blocking异步传输
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        # 使用普通精度训练
        prob = model(x)
        loss = loss_func(prob, y)
        loss.backward()
        optimizer.step()
        
        # 在GPU上直接计算准确率，避免频繁的CPU-GPU同步
        with torch.no_grad():
            pred = torch.argmax(prob, dim=1)
            batch_correct = (pred == y).sum().item()
            
            total_loss += loss.item() * x.size(0)
            total_correct += batch_correct
            total_samples += x.size(0)
            
            # 每50个batch更新一次显示信息
            if batch % 50 == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                avg_acc = total_correct / total_samples if total_samples > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                data.set_postfix_str(
                    f"loss:{avg_loss:.4f}, acc:{avg_acc:.3f}, lr:{current_lr:.7f}"
                )
    
    scheduler.step()
    
    # 计算epoch的最终指标
    epoch_loss = total_loss / total_samples if total_samples > 0 else 0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return epoch_acc, epoch_loss


def get_val_result(model, val_loader, loss_func, device):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # 使用更简洁的循环，避免频繁的numpy转换
        for x, y in tqdm(val_loader, desc="Validation", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            prob = model(x)
            loss = loss_func(prob, y)
            
            pred = torch.argmax(prob, dim=1)
            batch_correct = (pred == y).sum().item()
            
            total_loss += loss.item() * x.size(0)
            total_correct += batch_correct
            total_samples += x.size(0)
    
    # 计算整体指标
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_acc, avg_loss


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name, early_stopped=False):
    plt.figure(figsize=(12, 5))
    
    # 准确率图
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, "r-", label="Train", linewidth=2)
    plt.plot(epochs, val_accs, "g-", label="Val", linewidth=2)
    
    # 标记最佳准确率点
    best_epoch = val_accs.index(max(val_accs)) + 1
    plt.scatter(best_epoch, max(val_accs), color='blue', s=100, zorder=5, 
                label=f'Best Val Acc: {max(val_accs):.4f}')
    
    # 如果早停了，标记早停点
    if early_stopped and len(epochs) < 20:  # 假设总epochs为20
        plt.axvline(x=len(epochs), color='gray', linestyle='--', alpha=0.7, label='Early Stop')
    
    plt.title(f"{model_name} - Accuracy vs Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 损失图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, "r-", label="Train", linewidth=2)
    plt.plot(epochs, val_losses, "g-", label="Val", linewidth=2)
    
    # 如果早停了，标记早停点
    if early_stopped and len(epochs) < 20:
        plt.axvline(x=len(epochs), color='gray', linestyle='--', alpha=0.7, label='Early Stop')
    
    plt.title(f"{model_name} - Loss vs Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 根据是否早停调整文件名
    suffix = "_early_stopped" if early_stopped else ""
    plt.savefig(f"images/{model_name}_epoch_acc_loss{suffix}.jpg", dpi=150, bbox_inches='tight')
    plt.close()


def save_training_history(train_accs, train_losses, val_accs, val_losses, model_name):
    """保存训练历史到Excel文件"""
    df_acc_loss = pd.DataFrame({
        "epoch": range(1, len(train_accs) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    })
    
    # 添加最佳epoch标记
    best_epoch = val_accs.index(max(val_accs)) + 1
    df_acc_loss["is_best"] = df_acc_loss["epoch"] == best_epoch
    
    df_acc_loss.to_excel(f"results/{model_name}_epoch_acc_loss.xlsx", index=False)
    print(f"训练历史已保存到: results/{model_name}_epoch_acc_loss.xlsx")


def print_training_summary(model_name, total_epochs, max_val_acc, best_epoch, 
                          final_train_acc, final_val_acc, early_stopped=False):
    """打印训练摘要"""
    print("\n" + "="*60)
    print(f"训练摘要 - {model_name}")
    print(f"{'='*60}")
    print(f"训练状态: {'早停' if early_stopped else '正常完成'}")
    print(f"总训练轮次: {total_epochs}")
    print(f"最佳验证准确率: {max_val_acc:.4f} (epoch {best_epoch})")
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最终验证准确率: {final_val_acc:.4f}")
    print(f"过拟合程度: {final_train_acc - final_val_acc:.4f}")
    print("="*60)
    
    # 保存摘要到文件
    summary = f"""训练摘要 - {model_name}
{'='*50}
训练状态: {'早停' if early_stopped else '正常完成'}
总训练轮次: {total_epochs}
最佳验证准确率: {max_val_acc:.4f} (epoch {best_epoch})
最终训练准确率: {final_train_acc:.4f}
最终验证准确率: {final_val_acc:.4f}
过拟合程度: {final_train_acc - final_val_acc:.4f}
{'='*50}
"""
    
    with open(f"results/{model_name}_training_summary.txt", "w") as f:
        f.write(summary)


def resume_training(model_name, checkpoint_path=None, total_epochs=50):
    """恢复训练的主函数"""
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(model_name)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n{'='*60}")
        print(f"从checkpoint恢复训练")
        print(f"Checkpoint文件: {checkpoint_path}")
        print(f"{'='*60}")
        
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32计算（如果GPU支持）
    
    # 训练参数
    MODEL_NAME = "lyj_swin-t"  
    MODEL_NAME = "swin-t"
    TOTAL_EPOCHS = 50
    CHECKPOINT_INTERVAL = 5  # 每5个epoch保存一次checkpoint
    
    # 检查是否要恢复训练
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--resume":
            checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
            resume_training(MODEL_NAME, checkpoint_path, TOTAL_EPOCHS)
        elif sys.argv[1] == "--train":
            train(MODEL_NAME, total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
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
            train(MODEL_NAME, total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)
        elif choice == "2":
            resume_training(MODEL_NAME, total_epochs=TOTAL_EPOCHS)
        elif choice == "3":
            checkpoint_path = input("请输入checkpoint文件路径: ").strip()
            resume_training(MODEL_NAME, checkpoint_path, TOTAL_EPOCHS)
        else:
            print("无效选项，开始新训练...")
            train(MODEL_NAME, total_epochs=TOTAL_EPOCHS, checkpoint_interval=CHECKPOINT_INTERVAL)