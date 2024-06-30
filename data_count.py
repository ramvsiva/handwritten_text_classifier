import pandas as pd


def count_samples_per_class(csv_file):
    # Load CSV file
    data = pd.read_csv(csv_file)

    # Count how many samples have a '1' for each class
    class_counts = data.drop('filename', axis=1).sum(axis=0)

    return class_counts


# Specify the path to your CSV file
csv_file = 'image_labels.csv'

# Get counts of samples per class
class_counts = count_samples_per_class(csv_file)

# Print the counts
print("Number of samples per class:")
print(class_counts)
