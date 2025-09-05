import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from util.box_ops import box_xyxy_to_cxcywh  # 从DETR官方仓库导入


class DotaDataset(Dataset):
    def __init__(self, root, split='train', transforms=None, skip_difficult=True):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split, 'images')
        self.label_dir = os.path.join(root, split, 'labelTxt')
        self.transforms = transforms
        self.skip_difficult = skip_difficult  # 是否跳过difficult=1的样本
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]

        # DOTA类别到ID的映射（共15类）
        self.category2id = {
            'plane': 0, 'ship': 1, 'storage-tank': 2,
            'baseball-diamond': 3, 'tennis-court': 4,
            'basketball-court': 5, 'ground-track-field': 6,
            'harbor': 7, 'bridge': 8, 'large-vehicle': 9,
            'small-vehicle': 10, 'helicopter': 11, 'roundabout': 12,
            'soccer-ball-field': 13, 'swimming-pool': 14
        }

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))

        # 读取图像并获取尺寸
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        # 解析labelTxt文件（旋转框→轴对齐框）
        anns = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                parts = line.split()
                if len(parts) != 10:  # 确保每行有10个字段（8坐标+类别+难度）
                    continue

                # 提取旋转框的4个顶点坐标（x1,y1,x2,y2,x3,y3,x4,y4）
                coords = list(map(float, parts[:8]))
                xs = coords[0::2]  # [x1, x2, x3, x4]
                ys = coords[1::2]  # [y1, y2, y3, y4]

                # 计算轴对齐框的坐标（xmin, ymin, xmax, ymax）
                xmin, ymin = min(xs), min(ys)
                xmax, ymax = max(xs), max(ys)

                # 提取类别和难度标记
                category = parts[8]
                difficult = int(parts[9])

                # 跳过difficult=1的样本（可选）
                if self.skip_difficult and difficult == 1:
                    continue

                # 检查类别是否有效
                if category not in self.category2id:
                    continue  # 忽略未识别的类别

                anns.append({
                    'bbox': [xmin, ymin, xmax, ymax],  # 轴对齐框（xyxy格式）
                    'category_id': self.category2id[category]
                })

        # 构建DETR需要的target字典
        if not anns:  # 无有效标注时返回空目标（避免训练报错）
            target = {
                "image_id": torch.tensor([idx]),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                "image_id": torch.tensor([idx]),
                "boxes": torch.as_tensor([ann['bbox'] for ann in anns], dtype=torch.float32),  # [xmin,ymin,xmax,ymax]
                "labels": torch.as_tensor([ann['category_id'] for ann in anns], dtype=torch.int64),
                "area": torch.as_tensor(
                    [(ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1]) for ann in anns],
                    dtype=torch.float32),
                "iscrowd": torch.zeros(len(anns), dtype=torch.int64)
            }

        # 应用数据增强（官方默认增强，如翻转、颜色抖动等）
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # 坐标转换：xyxy → cxcywh（归一化到[0,1]）
        target["boxes"] = box_xyxy_to_cxcywh(target["boxes"])
        target["boxes"] /= torch.tensor([img_width, img_height, img_width, img_height])

        return image, target

    def __len__(self):
        return len(self.img_names)
