import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

# ===================== 核心配置（和你原有Stage2一致） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "aligned_faces train"  # A-S训练集目录（和你一致）
PRETRAIN_CKPT = "cls_pretrain_ckpt/cls_pretrain_epoch120.pth"  # Stage1预训练权重
SAVE_DIR = "facenet_final_ckpt_arcface"  # Stage2权重保存目录
NUM_CLASSES = 93  # 总类别数（和你一致）
FEAT_DIM = 128  # 特征维度（和你一致）

# 训练超参
EPOCHS = 80  # 训练轮数（建议50-80轮）
BATCH_SIZE = 32
LR = 1e-4  # 学习率（和你原有一致）
WEIGHT_DECAY = 5e-4
LOG_INTERVAL = 10  # 日志打印间隔
SAVE_INTERVAL = 10  # 权重保存间隔

# 固定随机种子（保证可复现）
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ===================== 1. ArcFace Loss 核心类（新增） =====================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features=FEAT_DIM, out_features=NUM_CLASSES, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 尺度因子（固定30，ArcFace默认）
        self.m = m  # 角度边际（固定0.5，ArcFace默认）

        # 类别中心权重矩阵（ArcFace核心）
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算角度参数（避免训练中重复计算）
        self.cos_m = torch.tensor(np.cos(m)).to(DEVICE)
        self.sin_m = torch.tensor(np.sin(m)).to(DEVICE)
        self.th = torch.tensor(np.cos(np.pi - m)).to(DEVICE)
        self.mm = torch.tensor(np.sin(np.pi - m) * m).to(DEVICE)

    def forward(self, feature, label):
        # Step1: L2归一化（特征+权重，ArcFace必须）
        feature_norm = nn.functional.normalize(feature, p=2, dim=1)
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)

        # Step2: 计算余弦相似度
        cosine = nn.functional.linear(feature_norm, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Step3: 加角度边际（ArcFace核心，强制拉开类间距离）
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Step4: 构建标签掩码（仅对正确类别应用边际）
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Step5: 最终logits（尺度缩放增强区分度）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Step6: 交叉熵损失（和原有分类损失兼容，易收敛）
        loss = nn.functional.cross_entropy(output, label)
        return loss


# ===================== 2. 模型定义（和你原有Stage2一致） =====================
class FaceNetCls(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        from torchvision import models
        # 骨干网络：ResNet34（和你原有一致）
        self.backbone = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # 冻结骨干网络（仅训练特征层和分类层，和你原有一致）
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 特征编码层（128维，和你原有一致）
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(2048, FEAT_DIM),
            nn.LayerNorm(FEAT_DIM)
        )

    def forward(self, x):
        # 骨干网络提取特征
        x = self.backbone(x)
        # 编码为128维特征
        feat = self.embedding(x)
        # L2归一化（ArcFace必须，和你原有一致）
        feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat


# ===================== 3. 数据集加载（和你原有Stage2一致） =====================
class FaceTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.samples = []

        # 构建类别映射
        person_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for idx, person in enumerate(person_dirs):
            clean_name = person.replace("pins_", "")
            self.class_to_idx[clean_name] = idx
            self.idx_to_class[idx] = clean_name

            # 加载该人的所有图片
            img_paths = glob.glob(os.path.join(root_dir, person, "*.jpg")) + glob.glob(
                os.path.join(root_dir, person, "*.png"))
            for img_path in img_paths:
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ===================== 4. 数据预处理（和你原有Stage2一致） =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 轻微数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 构建数据集和数据加载器
train_dataset = FaceTrainDataset(TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4 if os.name != "nt" else 0,  # Windows下num_workers=0
    pin_memory=True
)


# ===================== 5. 模型初始化（加载Stage1预训练权重） =====================
def init_model():
    # 初始化模型
    model = FaceNetCls(num_classes=NUM_CLASSES).to(DEVICE)

    # 加载Stage1预训练权重（和你原有一致）
    if os.path.exists(PRETRAIN_CKPT):
        print(f"🔧 加载Stage1预训练权重：{PRETRAIN_CKPT}")
        pretrain_dict = torch.load(PRETRAIN_CKPT, map_location=DEVICE)
        model_dict = model.state_dict()

        # 过滤不匹配的权重（仅加载骨干+embedding）
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print(f"✅ 加载{len(pretrain_dict)}个权重参数")
    else:
        print(f"⚠️ 未找到Stage1预训练权重：{PRETRAIN_CKPT}")

    return model


# ===================== 6. 训练主逻辑（仅替换损失函数） =====================
def train_stage2():
    # 初始化模型
    model = init_model()

    # 定义损失函数（核心修改：替换为ArcFaceLoss）
    criterion = ArcFaceLoss(in_features=FEAT_DIM, out_features=NUM_CLASSES).to(DEVICE)

    # 定义优化器（和你原有一致）
    optimizer = optim.Adam([
        {'params': model.embedding.parameters()},
        {'params': criterion.weight}  # 优化ArcFace的类别中心
    ], lr=LR, weight_decay=WEIGHT_DECAY)

    # 学习率调度器（可选，提升收敛效果）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 创建权重保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 开始训练
    print(f"\n🚀 开始Stage2训练（ArcFace Loss）")
    print(f"📊 训练集样本数：{len(train_dataset)}")
    print(f"⚙️ 设备：{DEVICE} | 批次大小：{BATCH_SIZE} | 训练轮数：{EPOCHS}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            feats = model(imgs)
            loss = criterion(feats, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            total_loss += loss.item()
            # 计算预测类别（ArcFace的logits需重新计算）
            cosine = nn.functional.linear(
                nn.functional.normalize(feats, p=2, dim=1),
                nn.functional.normalize(criterion.weight, p=2, dim=1)
            )
            _, predicted = torch.max(cosine * criterion.s, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 打印批次日志
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })

        # 学习率调度
        scheduler.step()

        # 计算本轮指标
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"\n📈 Epoch {epoch} 总结：")
        print(f"   平均损失：{epoch_loss:.4f} | 训练准确率：{epoch_acc:.2f}%")

        # 保存权重
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(SAVE_DIR, f"facenet_embed_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 保存权重：{ckpt_path}")

    print("\n🏁 Stage2训练完成！")
    print(f"📌 最终权重保存至：{SAVE_DIR}")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    train_stage2()