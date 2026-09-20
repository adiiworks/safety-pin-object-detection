import os
import torch
import cv2
import json
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Device Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_trained_model(weights_path="model_weights.pth"):
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None).backbone
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Model Load
model = load_trained_model("model_weights.pth")

# Multiple Images Processing Function with JSON Output
def process_image_folder(input_folder="test_images", output_folder="results", confidence_threshold=0.5):
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f" Folder '{input_folder}' created. Please input images!")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Folder me se saari images ki list lein
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if len(image_files) == 0:
        print(f" '{input_folder}' folder me koi image nahi mili!")
        return

    print(f"Total {len(image_files)} Found. Inference starting...\n" + "-"*40)

    # Dictionary to hold JSON results
    json_results = {
        "metadata": {
            "total_images_processed": len(image_files),
            "confidence_threshold": confidence_threshold
        },
        "images": []
    }

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            continue

        img_height, img_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(img_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        image_detections = []
        box_count = 0

        for box, score in zip(boxes, scores):
            if score >= confidence_threshold:
                box_count += 1
                x1, y1, x2, y2 = map(int, box)

                # Append bounding box data to list
                image_detections.append({
                    "box_id": box_count,
                    "label": "green_box",
                    "confidence": round(float(score), 4),
                    "bbox": [x1, y1, x2, y2]  # [xmin, ymin, xmax, ymax]
                })

                # Draw Bounding Box & Label Text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image, f"green_box {score:.2f}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Count Overlay on Output Image
        cv2.putText(image, f"Total Box: {box_count}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # Save annotated image
        save_path = os.path.join(output_folder, f"predicted_{filename}")
        cv2.imwrite(save_path, image)

        # Append per-image result for JSON
        json_results["images"].append({
            "filename": filename,
            "image_size": {"width": img_width, "height": img_height},
            "total_green_boxes": box_count,
            "detections": image_detections
        })
        
        print(f"[{idx}/{len(image_files)}] {filename} -> Count: {box_count} | Saved to: {save_path}")

    # Save JSON results file
    json_path = os.path.join(output_folder, "detection_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)

    print("-" * 40)
    print(f"✅ JSON results successfully saved to: {json_path}")
    print(" All IMGs processed!")

# Run on Folder
process_image_folder("test_images", "results")