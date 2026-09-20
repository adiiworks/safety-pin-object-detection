import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 1. Load Fine-Tuned Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_optimized_model(num_classes=2):
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None).backbone
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

model = get_optimized_model(num_classes=2)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()

# 2. Setup Evaluation Metric
metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=[0.5])

test_img_dir = 'dataset/images/test'
test_xml_dir = 'dataset/labels/test'

preds, targets = [], []

print("Running Evaluation on Fine-Tuned Model...")
for img_name in sorted(os.listdir(test_img_dir)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(test_img_dir, img_name)
    xml_path = os.path.join(test_xml_dir, os.path.splitext(img_name)[0] + '.xml')

    gt_boxes, gt_labels = [], []
    if os.path.exists(xml_path):
        root = ET.parse(xml_path).getroot()
        for member in root.findall('object'):
            if member.find('name').text == 'green_box':
                gt_labels.append(1)
                bndbox = member.find('bndbox')
                gt_boxes.append([
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)
                ])

    gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32).to(device) if len(gt_boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32).to(device)
    gt_labels_t = torch.tensor(gt_labels, dtype=torch.int64).to(device) if len(gt_labels) > 0 else torch.zeros((0,), dtype=torch.int64).to(device)

    image = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    preds.append({
        'boxes': prediction['boxes'].detach().cpu(),
        'scores': prediction['scores'].detach().cpu(),
        'labels': prediction['labels'].detach().cpu()
    })

    targets.append({
        'boxes': gt_boxes_t.cpu(),
        'labels': gt_labels_t.cpu()
    })

metric.update(preds, targets)
results = metric.compute()

map_50 = results['map_50'].item()
recall = results['mar_100'].item()
f1_score = 2 * (map_50 * recall) / (map_50 + recall + 1e-6)

print("\n================ UPDATED FINE-TUNED METRICS ================")
print(f"mAP @ IoU 0.50 : {map_50:.4f}  ({map_50*100:.2f}%)")
print(f"Recall        : {recall:.4f}  ({recall*100:.2f}%)")
print(f"F1-Score      : {f1_score:.4f}  ({f1_score*100:.2f}%)")
print("=============================================================")