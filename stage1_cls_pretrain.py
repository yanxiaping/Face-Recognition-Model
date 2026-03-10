"""分类预训练"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from read_data import MyData
# ===================== 基础配置 =====================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 阶段1超参数（分类预训练）
EPOCHS_CLS = 70
BATCH_SIZE = 64
LR_CLS = 3e-4
# 数据路径（A-S是训练集，93类）
TRAIN_DIR = "aligned_faces train"  # 你的A-S类人脸目录
# 保存预训练模型的路径
CKPT_DIR = "cls_pretrain_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)



# 加载分类训练数据集
train_dataset = MyData(root_dir="aligned_faces train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ===================== 2. 分类预训练模型（含128维嵌入+93类分类头） =====================
class FaceNetCls(nn.Module):
    def __init__(self, num_classes=93):
        super().__init__()
        # 主干：ResNet34（特征提取）
        self.backbone = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 冻结主干（先训分类头，可选后续解冻微调）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 核心：128维嵌入层（后续要保留的部分）
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(2048, 128),
            nn.LayerNorm(128)  # 归一化，为后续对比损失做准备
        )

        # 分类头（阶段1用，阶段2删除）
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # 93类分类
        )

    def forward(self, x):
        # 主干特征提取
        x = self.backbone(x)
        # 128维嵌入
        feat = self.embedding(x)
        # 分类输出
        cls_out = self.classifier(feat)
        return cls_out

    # 专门提取128维特征的方法（阶段2用）
    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            x = self.backbone(x)
            feat = self.embedding(x)
            # L2归一化（FaceNet标准）
            feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat


# ===================== 3. 分类预训练训练逻辑 =====================
# 初始化模型
model_cls = FaceNetCls(num_classes=93).to(DEVICE)
pretrained_ckpt = os.path.join(CKPT_DIR, "cls_pretrain_epoch50.pth")
if os.path.exists(pretrained_ckpt):
    model_cls.load_state_dict(torch.load(pretrained_ckpt, map_location=DEVICE))
    print(f"✅ 加载50轮权重，续训到{EPOCHS_CLS}轮")
# 损失：交叉熵（分类任务）
criterion_cls = nn.CrossEntropyLoss()
# 优化器：只训嵌入层+分类头（主干冻结）
optimizer = optim.Adam(
    list(model_cls.embedding.parameters()) + list(model_cls.classifier.parameters()),
    lr=LR_CLS,
    weight_decay=1e-5
)


# 开始分类预训练
def train_classifier():
    model_cls.train()
    model_cls.backbone.eval()

    # 新增：判断是否是续训，确定起始轮数
    start_epoch = 0
    if os.path.exists(pretrained_ckpt):
        start_epoch = 50  # 已有50轮，续训从第51轮开始

    for epoch in range(EPOCHS_CLS):
        # 计算真实的当前轮数
        real_epoch = start_epoch + epoch + 1
        # 计算总轮数（如果是续训，总轮数就是 start_epoch + EPOCHS_CLS）
        total_epochs = start_epoch + EPOCHS_CLS

        total_loss = 0.0
        correct = 0
        total = 0
        # 修改进度条描述，显示真实轮数
        pbar = tqdm(train_loader, desc=f"Cls Epoch {real_epoch}/{total_epochs}")

        for img, label in pbar:
            img, label = img.to(DEVICE), label.to(DEVICE)
            cls_out = model_cls(img)
            loss = criterion_cls(cls_out, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_cls.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            _, pred = cls_out.max(1)
            correct += pred.eq(label).sum().item()
            total += img.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / total
        acc = 100 * correct / total
        # 修改打印的轮数
        print(f"Epoch {real_epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # 修改保存的文件名，使用真实轮数
        if real_epoch % 10 == 0:
            torch.save(model_cls.state_dict(), os.path.join(CKPT_DIR, f"cls_pretrain_epoch{real_epoch}.pth"))

# 执行分类预训练（这一步跑完，模型就学会区分93类人脸了）
if __name__ == "__main__":

    train_classifier()