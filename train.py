import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='path_to_save_model'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path to save the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
            model.save_pretrained(self.path)


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        labels = torch.tensor(self.img_labels.iloc[idx, 1:].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, labels


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = CustomDataset(csv_file='image_labels.csv', img_dir='training_images', transform=transform)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                  num_labels=12, # Specify your number of classes here
                                                  ignore_mismatched_sizes=True) # To ignore the original head size mismatch


print("New classifier weight shape:", model.classifier.weight.shape)
print("New classifier bias shape:", model.classifier.bias.shape)
model.to('cuda')


optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
early_stopping = EarlyStopping(patience=5, verbose=True, path='trained_models')

model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc="Training Epoch {:1d}".format(epoch + 1), leave=False, disable=False)
    for batch in progress_bar:
        batch = [item.to(device) for item in batch]
        inputs, labels = batch
        outputs = model(inputs)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix({'training_loss': loss.item()})

    model.eval()
    eval_loss = 0
    eval_steps = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = [item.to(device) for item in batch]
            inputs, labels = batch
            outputs = model(inputs)
            loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
            eval_loss += loss.item()
            eval_steps += 1

            preds = torch.sigmoid(outputs.logits).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    print(
        f'Epoch {epoch + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Avg Eval Loss: {eval_loss / eval_steps:.4f}')

    avg_eval_loss = eval_loss / eval_steps
    early_stopping(avg_eval_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Save the model if not saved by early stopping
if not early_stopping.early_stop:
    model_path = 'trained_vit_model'
    model.save_pretrained(model_path)
    print(f'Model saved to {model_path}')
