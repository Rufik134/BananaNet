import os
import csv
import random

# deficne root dataset folder
dataset_root = "dataset"

# day ranges for each ripeness class
class_day_ranges = {
    "Class A": (1, 6),
    "Class B": (7, 14),
    "Class C": (15, 22),
    "Class D": (23, 28)
}

# output CSV file
output_file = "banana_ripeness_ttl_dataset.csv"

# gather entries
entries = []

# traverse dataset
for split in ["train", "test", "validation"]:
    split_path = os.path.join(dataset_root, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        min_day, max_day = class_day_ranges.get(class_name, (0, 0))
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                estimated_day = random.randint(min_day, max_day)
                ttl = 28 - estimated_day
                entries.append({
                    "image_path": os.path.join(split, class_name, img_file),
                    "ripeness_class": class_name,
                    "estimated_day": estimated_day,
                    "ttl_days_left": ttl
                })

# write data to CSV
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["image_path", "ripeness_class", "estimated_day", "ttl_days_left"])
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry)

print(f"Dataset CSV saved as: {output_file}")
