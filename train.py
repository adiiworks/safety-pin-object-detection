import os
import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import random

# ==========================================
# 1. Dataset with Data Augmentation
# ==========================================
class AugmentedGreenBoxVOCDataset(Dataset):
    def __init__(self, root_dir='dataset', split='train', augment=False):
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'labels', split)
        self.augment = augment
        
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
        h, w, _ = image.shape

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

        # Convert Image to Tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Apply Augmentations during training
        if self.augment and len(boxes) > 0:
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                image_tensor = F.hflip(image_tensor)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            # 2. Random Brightness & Contrast Adjustment
            if random.random() > 0.5:
                image_tensor = F.adjust_brightness(image_tensor, brightness_factor=random.uniform(0.8, 1.2))
                image_tensor = F.adjust_contrast(image_tensor, contrast_factor=random.uniform(0.8, 1.2))

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image_tensor, target

    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 2. Setup Optimized Faster R-CNN Model
# ==========================================
def get_optimized_model(num_classes=2):
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").backbone
    
    # Custom Anchor Generator fine-tuned for dense & small green boxes
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    return model

# ==========================================
# 3. Training Loop (25 Epochs + LR Scheduler)
# ==========================================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Training using device: {device}")

# Datasets with Augmentation enabled for training
train_dataset = AugmentedGreenBoxVOCDataset(split='train', augment=True)
val_dataset = AugmentedGreenBoxVOCDataset(split='val', augment=False)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

model = get_optimized_model(num_classes=2).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Decay LR by factor of 0.1 at epoch 15
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

num_epochs = 25
os.makedirs("models", exist_ok=True)

print("\nStarting Extended Fine-Tuning (25 Epochs)...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1:02d}/{num_epochs}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

# Overwrite fine-tuned weights
torch.save(model.state_dict(), "models/model_weights.pth")
print("\nFine-Tuning Complete! Model saved as 'models/model_weights.pth'")