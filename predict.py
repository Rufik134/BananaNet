import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import urllib.request

#  create banana model
class BananaNet(nn.Module):
    def __init__(self):
        super(BananaNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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
        class_output = self.classifier(x)
        ttl_output = self.regressor(x)
        return class_output, ttl_output.squeeze(1)

# load BananaNet model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BananaNet().to(device)
model.load_state_dict(torch.load("banana_model.pth", map_location=device))
model.eval()

# mage preprocessing for BananaNet
banana_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# normalization parameters used during training
ttl_mean = 5.0
ttl_std = 2.0

# class mapping
class_names = ['Class A (Unripe)', 'Class B (Ripe)', 'Class C (Nearly Overripe)', 'Class D (Overripe)']

# load ImageNet class names 
imagenet_classes = []
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as url:
    imagenet_classes = [line.decode('utf-8').strip() for line in url]

# banana detection using ResNet -  this used to detect banan object itself
def is_banana(image, threshold=0.5):
    detector_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img_tensor = detector_transform(image).unsqueeze(0)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()

    with torch.no_grad():
        outputs = resnet(img_tensor)
        probs = F.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, dim=0)

    label = imagenet_classes[top_class.item()].lower()
    print(f"🔍 Detected object: {label} ({top_prob.item()*100:.1f}%)")
    
    # keywords for  flexible banana detection
    banana_keywords = ['banana', 'plantain']
    return any(keyword in label for keyword in banana_keywords) and top_prob.item() > threshold

# run prediction
def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    if not is_banana(image):
        print("❌ Not a banana (failed detection)")
        return None, None

    input_tensor = banana_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, ttl_norm = model(input_tensor)

    pred_class = torch.argmax(class_logits, dim=1).item()
    pred_ttl = ttl_norm.item() * ttl_std + ttl_mean  # denormalize ttl

    print(f"✅ Banana detected! Ripeness: {class_names[pred_class]}, TTL: {pred_ttl:.1f} days")  
    return class_names[pred_class], pred_ttl

# terminal usage  to detect
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    predict(args.image)
