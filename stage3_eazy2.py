import os
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

# ===================== 核心配置（仅改这4个） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_CKPT = "facenet_final_ckpt_arcface/facenet_embed_epoch80.pth"  # 阶段2模型
PRETRAIN_CKPT = "cls_pretrain_ckpt/cls_pretrain_epoch120.pth"  # 阶段1预训练模型
TEST_DIR = "aligned_faces test"  # T-Z测试集目录

# 比例划分配置
DB_RATIO = 0.7  # 70%建库
TEST_RATIO = 0.3  # 30%测试
MIN_IMGS = 5  # 最少5张才参与测试


# ===================== 模型类（修复权重加载） =====================
class FaceNetCls(torch.nn.Module):
    def __init__(self, num_classes=93):
        super().__init__()
        from torchvision import models
        self.backbone = models.resnet34(pretrained=True)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embedding = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 7 * 7, 2048),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 128),
            torch.nn.LayerNorm(128)
        )
        # 对齐训练时的classifier结构
        self.classifier = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        feat = self.embedding(x)
        return self.classifier(feat)


class FaceNetEmbed(torch.nn.Module):
    def __init__(self, pretrain_ckpt):
        super().__init__()
        cls_model = FaceNetCls(93)
        # 非严格加载权重（忽略classifier的微小不匹配）
        state_dict = torch.load(pretrain_ckpt, map_location=DEVICE)
        cls_model.load_state_dict(state_dict, strict=False)

        self.backbone = cls_model.backbone
        self.embedding = cls_model.embedding
        self.to(DEVICE)

    def forward(self, x):
        x = self.backbone(x)
        feat = self.embedding(x)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)  # L2归一化（关键！）
        return feat


# ===================== 全局变量 =====================
global_db = {}  # {姓名: [特征1, 特征2,...]}
encoder_model = None


# ===================== 核心工具函数 =====================
def load_encoder():
    """只加载一次模型"""
    global encoder_model
    if encoder_model is None:
        print(f"\n🔧 加载人脸编码模型：{ENCODER_CKPT}")
        encoder_model = FaceNetEmbed(PRETRAIN_CKPT)
        # 加载Stage2训练好的特征提取权重
        stage2_state_dict = torch.load(ENCODER_CKPT, map_location=DEVICE)
        encoder_model.load_state_dict(stage2_state_dict, strict=False)
        encoder_model.eval()
        print("✅ 模型加载完成")
    return encoder_model


def extract_feature(img_path, model):
    """提取单张图片的128维特征"""
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在：{img_path}")
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(img_tensor)
        return feat.cpu().numpy().squeeze()
    except Exception as e:
        print(f"❌ 提取{os.path.basename(img_path)}特征失败：{str(e)[:50]}")
        return None


def euclidean_dist(feat1, feat2):
    """计算欧式距离（越小越相似）"""
    return np.linalg.norm(feat1 - feat2)


# ===================== 构建自适应特征库 =====================
def build_adaptive_db():
    """自适应构建特征库（随机打乱+比例划分）"""
    global global_db
    if global_db:
        print(f"\nℹ️ 已加载特征库，共{len(global_db)}人")
        return global_db

    model = load_encoder()
    print(f"\n📝 构建T-Z测试集特征库（{DB_RATIO}:{TEST_RATIO}比例划分）")

    person_dirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    print(f"👥 发现T-Z测试集人数：{len(person_dirs)}")

    for person_dir in tqdm(person_dirs, desc="处理每个人"):
        person_name = person_dir.replace("pins_", "") if "pins_" in person_dir else person_dir
        full_dir = os.path.join(TEST_DIR, person_dir)

        # 获取该人的所有照片
        all_imgs = glob.glob(os.path.join(full_dir, "*.jpg")) + glob.glob(os.path.join(full_dir, "*.png"))
        total_imgs = len(all_imgs)

        if total_imgs < MIN_IMGS:
            print(f"\n⚠️ {person_name} 照片不足（{total_imgs}张），跳过")
            continue

        # 随机打乱（核心：保证公平性）
        random.shuffle(all_imgs)

        # 按比例划分建库
        db_size = int(total_imgs * DB_RATIO)
        db_imgs = all_imgs[:db_size]

        # 提取建库特征
        person_feats = []
        for img_path in tqdm(db_imgs, desc=f"提取{person_name}特征", leave=False):
            feat = extract_feature(img_path, model)
            if feat is not None:
                person_feats.append(feat)

        if len(person_feats) > 0:
            global_db[person_name] = person_feats
            print(f"\n✅ {person_name}：总照片{total_imgs}张 → 建库{len(db_imgs)}张 → 有效特征{len(person_feats)}个")
        else:
            print(f"\n⚠️ {person_name} 无有效特征，跳过")

    print(f"\n🎉 特征库构建完成！有效人数：{len(global_db)}")
    return global_db


# ===================== 批量测试（核心优化：最小距离匹配） =====================
def batch_test():
    """批量测试（业界标准：最小距离匹配）"""
    model = load_encoder()
    db = build_adaptive_db()
    if not db:
        print("❌ 特征库为空，无法测试")
        return

    print(f"\n📝 开始批量测试T-Z数据集（最小距离匹配）")
    total_correct = 0
    total_samples = 0
    person_results = []

    person_dirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    for person_dir in tqdm(person_dirs, desc="批量测试"):
        person_name = person_dir.replace("pins_", "") if "pins_" in person_dir else person_dir
        if person_name not in db:
            continue

        full_dir = os.path.join(TEST_DIR, person_dir)
        all_imgs = glob.glob(os.path.join(full_dir, "*.jpg")) + glob.glob(os.path.join(full_dir, "*.png"))
        total_imgs = len(all_imgs)

        if total_imgs < MIN_IMGS:
            continue

        # 随机打乱+比例划分测试集
        random.shuffle(all_imgs)
        db_size = int(total_imgs * DB_RATIO)
        test_imgs = all_imgs[db_size:]
        test_size = len(test_imgs)

        # 测试该人的样本
        person_correct = 0
        person_total = test_size
        for img_path in tqdm(test_imgs, desc=f"测试{person_name}", leave=False):
            test_feat = extract_feature(img_path, model)
            if test_feat is None:
                person_total -= 1
                continue

            # ========== 核心优化：最小距离匹配（业界标准） ==========
            min_dist = float('inf')
            pred_name = "Unknown"
            for name in db.keys():
                # 计算测试特征与该人所有库特征的最小距离（不是平均！）
                distances = [euclidean_dist(test_feat, feat) for feat in db[name]]
                current_min_dist = np.min(distances)  # 关键：取最小距离

                if current_min_dist < min_dist:
                    min_dist = current_min_dist
                    pred_name = name
            # ======================================================

            # 统计结果
            if pred_name == person_name:
                person_correct += 1
                total_correct += 1
            total_samples += 1

        # 计算该人准确率
        person_acc = person_correct / person_total if person_total > 0 else 0.0
        person_results.append({
            "name": person_name,
            "total_imgs": total_imgs,
            "test_imgs": person_total,
            "correct": person_correct,
            "acc": person_acc
        })
        print(f"\n✅ {person_name}：总照片{total_imgs}张 → 测试{person_total}张 → 准确率{person_acc:.2%}")

    # 输出最终结果
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n========== 批量测试最终结果（最小距离匹配） ==========")
    print(f"📊 参与测试人数：{len(person_results)}")
    print(f"📈 总测试样本数：{total_samples}")
    print(f"🎯 整体准确率：{overall_acc:.2%}（{total_correct}/{total_samples}）")

    # 详细统计
    print(f"\n【每个人的测试详情】")
    for res in person_results:
        print(f"  {res['name']}：总照片{res['total_imgs']}张 → 测试{res['test_imgs']}张 → 准确率{res['acc']:.2%}")


# ===================== 单张图片测试（最小距离+详细日志） =====================
def test_single_img():
    """交互式单张测试（最小距离匹配）"""
    model = load_encoder()
    db = build_adaptive_db()
    if not db:
        print("❌ 特征库为空，无法测试")
        return

    print(f"\n=====================================")
    print("🎯 单张图片测试模式（输入q退出）")
    print(f"📌 特征库包含：{list(db.keys())}")
    print("=====================================")

    while True:
        img_path = input("\n请输入测试图片路径：").strip()
        if img_path.lower() == 'q':
            print("👋 退出单张测试模式")
            break

        test_feat = extract_feature(img_path, model)
        if test_feat is None:
            continue

        # 最小距离匹配
        dist_dict = {}
        for name in db.keys():
            distances = [euclidean_dist(test_feat, feat) for feat in db[name]]
            dist_dict[name] = {
                "min": np.min(distances),  # 最小距离（核心）
                "avg": np.mean(distances)  # 平均距离（参考）
            }

        # 按最小距离排序
        sorted_res = sorted(dist_dict.items(), key=lambda x: x[1]["min"])
        pred_name = sorted_res[0][0]
        first_min = sorted_res[0][1]["min"]
        second_name = sorted_res[1][0] if len(sorted_res) > 1 else "无"
        second_min = sorted_res[1][1]["min"] if len(sorted_res) > 1 else 0.0

        # 计算置信度（基于最小距离差）
        confidence = 0.0
        if second_min - first_min > 1e-8:
            confidence = 1 - (first_min / second_min)

        # 输出详细结果
        print(f"\n========== 单张图片测试结果（最小距离匹配） ==========")
        print(f"📸 测试图片：{os.path.basename(img_path)}")
        print(f"🎯 预测姓名：{pred_name}")
        print(f"📏 最小距离：{first_min:.4f} | 平均距离：{sorted_res[0][1]['avg']:.4f}")
        print(f"🥈 第二名：{second_name}（最小距离：{second_min:.4f}）")
        print(f"🔍 置信度：{confidence:.2%}")

        # 前3名详情
        print(f"\n📊 前3名匹配结果：")
        for i in range(min(3, len(sorted_res))):
            name = sorted_res[i][0]
            min_d = sorted_res[i][1]["min"]
            avg_d = sorted_res[i][1]["avg"]
            print(f"  第{i + 1}名：{name} → 最小{min_d:.4f} | 平均{avg_d:.4f}")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    random.seed(42)  # 固定随机种子（结果可复现）

    print("=" * 60)
    print("🚀 T-Z测试集最终版（最小距离匹配+自适应划分）")
    print("=" * 60)

    # 1. 构建特征库
    build_adaptive_db()

    # 2. 批量测试（最小距离匹配）
    batch_test()

    # 3. 单张图片测试
    test_single_img()

    print("\n🏁 所有测试完成！")