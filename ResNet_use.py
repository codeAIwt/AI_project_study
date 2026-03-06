import torch
from PIL import Image
from torchvision import models, transforms

img = Image.open(r"C:\Users\文档\Desktop\mei.jpg")

preprocess = transforms.Compose([   #预处理
    transforms.Resize(256),         # 缩放到256x256
    transforms.CenterCrop(224),     
    transforms.ToTensor(),         
    transforms.Normalize(           # 标准化（ImageNet统计值）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#图片处理
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

#加载模型
model = models.resnet18(pretrained=True)
model.eval()

with torch.no_grad():
    features = model(input_batch)
    
print(f"特征维度为:{features.shape}")