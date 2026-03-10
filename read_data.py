import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyData(Dataset):
    def __init__(self,root_dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.samples.append([img_path, self.class_to_idx[cls_name]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path,cls = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        re_img = self.transform(img)
        # cls = torch.tensor(cls)
        # one_hot = torch.nn.functional.one_hot(cls, num_classes=len(self.classes))
        return re_img,cls


if __name__ =='__main__':
    root_dir = "aligned_faces train/"
    dataset = MyData(root_dir=root_dir) # 先实例化类
    img,cls = dataset[10000]
    print(cls)
    plt.imshow(img.permute(1,2,0))
    plt.show()
