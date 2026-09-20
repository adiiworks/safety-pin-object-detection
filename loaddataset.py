import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader

class GreenBoxVOCDataset(Dataset):
    def __init__(self, root_dir='dataset', split='train'):
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'labels', split)

        if os.path.exists(self.images_dir):
            self.image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            self.image_files = []

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        xml_path = os.path.join(self.labels_dir, os.path.splitext(img_filename)[0] + '.xml')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        if os.path.exists(xml_path):
            root = ET.parse(xml_path).getroot()
            for member in root.findall('object'):
                if member.find('name').text == 'green_box':
                    labels.append(1)
                    bndbox = member.find('bndbox')
                    boxes.append([
                        float(bndbox.find('xmin').text),
                        float(bndbox.find('ymin').text),
                        float(bndbox.find('xmax').text),
                        float(bndbox.find('ymax').text)
                    ])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image_tensor, target

    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    return tuple(zip(*batch))

print("Dataset Class Defined!")