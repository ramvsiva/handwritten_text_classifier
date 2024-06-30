import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification


def load_model(model_path, num_labels):
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=num_labels)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_image(model, processed_image):
    with torch.no_grad():
        outputs = model(processed_image.to(model.device))
        predictions = torch.sigmoid(outputs.logits).squeeze(0)  # Apply sigmoid and remove batch dimension
    return predictions


def print_class_predictions(predictions, threshold=0.5):
    class_names = [f'C{i}' for i in range(predictions.shape[0])]  # Assuming class names are like 'C0', 'C1', etc.
    predicted_labels = [class_names[i] for i, pred in enumerate(predictions) if pred > threshold]
    print("Predicted classes:", predicted_labels)


model_path = ''  # Specify the path to your trained model directory
num_labels = 12  # Specify the correct number of classes
image_path = ''  # Specify the path to your test image

model = load_model(model_path, num_labels)
processed_image = process_image(image_path)
predictions = predict_image(model, processed_image)
print_class_predictions(predictions)
