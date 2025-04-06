import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model import TowerClassifier
import numpy as np

class TowerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['guyed', 'lattice', 'monopole', 'water_tank']
        self.images = []
        self.labels = []
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        
        # Load images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            print(f"\nLoading images from {class_dir}:")
            class_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.startswith('.'): 
                    continue  # Skip hidden files
                
                # Check file extension
                ext = os.path.splitext(img_name.lower())[1]
                if ext not in self.valid_extensions:
                    print(f"Skipping {img_name} - unsupported format")
                    continue
                
                img_path = os.path.join(class_dir, img_name)
                if os.path.exists(img_path):
                    try:
                        # Try to open the image to verify it's valid
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's actually an image
                        self.images.append(img_path)
                        self.labels.append(class_idx)
                        class_count += 1
                    except Exception as e:
                        print(f"Error verifying {img_name}: {str(e)}")
            
            print(f"Successfully loaded {class_count} images for class '{class_name}'")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random valid image instead
            valid_idx = np.random.randint(len(self))
            while valid_idx == idx:  # Make sure we don't pick the same problematic image
                valid_idx = np.random.randint(len(self))
            return self.__getitem__(valid_idx)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
print("\nInitializing dataset...")
dataset = TowerDataset('training_data', transform=transform)
print(f"\nTotal images found for training: {len(dataset)}")

if len(dataset) == 0:
    print("Error: No valid images found for training!")
    exit(1)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = TowerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
print("\nStarting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print batch progress
        if batch_idx % 5 == 0:
            print(f'Batch [{batch_idx}/{len(dataloader)}]', end='\r')
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

print("\nTraining finished!")

# Save the model
torch.save(model.state_dict(), 'tower_classifier.pth')
print("Model saved as tower_classifier.pth") 