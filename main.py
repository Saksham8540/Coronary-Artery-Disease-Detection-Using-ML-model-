import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = transform_base(image).unsqueeze(0)  # [1, 3, 224, 224]
    return image.to(device)

# Load Model
def load_model(weights_path='best_model.pth'):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Prediction function
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0
    return prediction, prob

# Train function
def train_model(train_dir='dataset/trainset', test_dir='dataset/testset', num_epochs=20):
    # Augmentations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    full_train_data = datasets.ImageFolder(train_dir, transform=transform_train)
    test_data = datasets.ImageFolder(test_dir, transform=transform_base)

    # Train/Val Split
    val_size = int(0.2 * len(full_train_data))
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Handle class imbalance
    label_counts = Counter([full_train_data[i][1] for i in range(len(full_train_data))])
    pos_weight = torch.tensor([label_counts[0] / label_counts[1]]).to(device)

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    def evaluate(loader, title="Validation"):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"{title} Accuracy: {acc:.2f}%")
        return acc

    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
        val_acc = evaluate(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ Best model saved.\n")

    # Final Test Evaluation
    print("📊 Final evaluation on test set:")
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate(test_loader, title="Test")

# Entry point
if __name__ == '__main__':
    print("🚀 Starting training...")
    train_model()