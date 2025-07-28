import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import BananaDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

# using device GPU if aviable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load dataset
dataset = BananaDataset("banana_ripeness_ttl_dataset.csv", "dataset", transform=transform)

# split our training 80 to 20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# model
class BananaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, 4)
        self.regressor = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.classifier(x), self.regressor(x).squeeze(1)

model = BananaNet().to(device)

# loss and optimizer
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training process
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_class_loss, total_reg_loss = 0, 0

    for images, labels, ttls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels, ttls = images.to(device), labels.to(device), ttls.float().to(device)

        optimizer.zero_grad()
        pred_class, pred_ttl = model(images)

        loss_class = criterion_class(pred_class, labels)
        loss_reg = criterion_reg(pred_ttl, ttls)
        total_loss = loss_class + loss_reg

        total_loss.backward()
        optimizer.step()

        total_class_loss += loss_class.item()
        total_reg_loss += loss_reg.item()

    print(f"Epoch {epoch+1}: Classification Loss = {total_class_loss:.4f}, TTL Loss = {total_reg_loss:.4f}")

# evaulation of validation
model.eval()
all_preds, all_labels = [], []
all_pred_ttls, all_true_ttls = [], []

with torch.no_grad():
    for images, labels, ttls in val_loader:
        images, labels, ttls = images.to(device), labels.to(device), ttls.float().to(device)
        pred_class, pred_ttl = model(images)

        all_preds.extend(torch.argmax(pred_class, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_pred_ttls.extend(pred_ttl.cpu().numpy())
        all_true_ttls.extend(ttls.cpu().numpy())

# metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

mae = mean_absolute_error(all_true_ttls, all_pred_ttls)
rmse = mean_squared_error(all_true_ttls, all_pred_ttls) ** 0.5

# print of results
print("\n📊 Validation Results:")
print(f"📌 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
print(f"📉 TTL Regression — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# saving results
torch.save(model.state_dict(), "banana_model.pth")
print("✅ Model saved as banana_model.pth")

# plot learning rate
epochs = list(range(1, num_epochs + 1))
initial_lr = 0.001
lrs = [initial_lr * (0.1 ** (epoch // 5)) for epoch in epochs]

plt.figure(figsize=(8, 5))
plt.plot(epochs, lrs, marker='o')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.savefig("learning_rate_schedule.png")
plt.show()
