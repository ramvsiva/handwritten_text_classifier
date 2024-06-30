import os
import shutil
import pandas as pd
from PIL import Image
import re


def extract_classes(filename):
    class_pattern = re.compile(r'C(\d+)')
    class_indices = [int(match.group(1)) for match in class_pattern.finditer(filename)]
    return class_indices


def create_label_vector(class_indices, num_classes=11):
    label_vector = [0] * num_classes
    for index in class_indices:
        label_vector[index] = 1
    return label_vector


def process_directory(source_directory, target_directory, csv_filename):
    data = []
    num_classes = 12  # This should cover C0 to C10; adjust if more classes exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    image_id = 0
    for subdir, dirs, files in os.walk(source_directory):
        for file in files:
            if file.startswith('written') and file.endswith('.png'):
                source_path = os.path.join(subdir, file)
                unique_filename = f"{image_id:08d}_{file}"
                target_path = os.path.join(target_directory, unique_filename)

                shutil.copy2(source_path, target_path)

                class_indices = extract_classes(file)
                label_vector = create_label_vector(class_indices, num_classes)

                data.append([unique_filename] + label_vector)

                image_id += 1

    columns = ['filename'] + [f'C{i}' for i in range(num_classes)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")
    print(f"Images copied to: {target_directory}")


source_directory = 'berlin/'
target_directory = 'training_images/'
csv_filename = 'image_labels.csv'

process_directory(source_directory, target_directory, csv_filename)
