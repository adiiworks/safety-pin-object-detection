import os
import shutil
import zipfile

# 1. Zip extract karein agar dataset.zip exist karta hai
if os.path.exists('dataset.zip'):
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('./temp_dataset')
    print("✅ Successfully extracted dataset.zip to temp folder")

# Target folders ready karein
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f'dataset/images/{split}', exist_ok=True)
    os.makedirs(f'dataset/labels/{split}', exist_ok=True)

# 2. Poore project / temp folder me se XML aur Image files traverse karke place karein
search_dirs = ['./temp_dataset', './images', './labels']

valid_img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

for search_dir in search_dirs:
    if os.path.exists(search_dir):
        for root, _, files in os.walk(search_dir):
            # Determine split (train/val/test) path me se
            root_lower = root.lower()
            target_split = 'train'
            if 'val' in root_lower:
                target_split = 'val'
            elif 'test' in root_lower:
                target_split = 'test'
            
            for file in files:
                src_path = os.path.join(root, file)
                
                # Agar image file hai
                if file.endswith(valid_img_exts):
                    dst_path = os.path.join(f'dataset/images/{target_split}', file)
                    shutil.move(src_path, dst_path)
                
                # Agar XML annotation file hai
                elif file.lower().endswith('.xml'):
                    dst_path = os.path.join(f'dataset/labels/{target_split}', file)
                    shutil.move(src_path, dst_path)

# Cleanup temporary folder
if os.path.exists('./temp_dataset'):
    shutil.rmtree('./temp_dataset', ignore_errors=True)

# 3. Final Verification
print("\n=== DATASET VERIFICATION ===")
for split in splits:
    img_dir = f'dataset/images/{split}'
    lbl_dir = f'dataset/labels/{split}'
    
    img_count = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    lbl_count = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
    
    print(f" - {split.capitalize():<5} -> Images: {img_count:<3} | Annotations: {lbl_count}")