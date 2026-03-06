from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
import os
import re

#加载文本模型
model_name = r'.\bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model0 = AutoModel.from_pretrained(model_name)

text_path = r'C:\Users\文档\Desktop\torchCS.txt'
with open(text_path, 'r', encoding='utf-8') as f:#打开文本，得到文件对象，并指定编码方式为UTF-8
    text = f.read()#读取文本，得到文本内容
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# 处理文本，得到输入张量，并进行填充和截断，
#return_tensors="pt"表示返回PyTorch张量，padding=True表示进行填充，truncation=True表示进行截断



from PIL import Image
from torchvision import models, transforms

img = Image.open(r"C:\Users\文档\Desktop\mei.jpg")

preprocess = transforms.Compose([   #预处理，将图片转换为模型输入的格式，
                                    #包括缩放、裁剪、转换为张量、标准化等操作
    transforms.Resize(128),         # 将图像缩放为128x128的大小
    transforms.CenterCrop(224),     # 图像的中心裁剪出一个224x224的区域，
                                        # 这是因为ResNet模型通常需要输入224x224的图像
    transforms.ToTensor(),          # 转换为张量，张量：是一种多维数组，
                                        # 用于存储和处理图像数据，
                                        # 张量的形状通常是(C, H, W)，C是通道数，H是高度，W是宽度，
    transforms.Normalize(           # 标准化（ImageNet统计值）
        mean=[0.485, 0.456, 0.406], # 均值
        std=[0.229, 0.224, 0.225]   # 标准差
    )
])

#图片处理
input_tensor = preprocess(img)#使用preprocess对图片预处理，得到输入张量
input_batch = input_tensor.unsqueeze(0)#在第0维添加一个维度，得到输入批次，
                                        #因为模型的输入通常是一个批次的图像，
                                        #所以需要将输入张量转换为一个批次的张量

#加载图片模型
model1 = models.resnet18(pretrained=True)#加载预训练的ResNet18模型，
                                        #pretrained=True表示加载预训练的权重
model1.eval()        #将模型设置为评估模式，
                      #在评估模式下，模型的参数不会被更新


with torch.no_grad():
    features_text = model0(**inputs).pooler_output
    features_image = model1(input_batch)

# print("\n=== text原始输出结构 ===")
# print(features_text)

# print("\n=== 句子特征向量 ===")
# print(f"text特征维度:{features_text.pooler_output.shape}")
# # print(f"text前5个数值:{features_text.pooler_output[0][:5].numpy()}")

# print(f"image特征维度为:{features_image.shape}")

##===实现联合和'多模态感知'===##

import torch.nn as nn
txt_dim = features_text.size(1)
img_dim = features_image.size(1)

text_proj = nn.Linear(txt_dim, 128)
image_proj = nn.Linear(img_dim, 128)

text_feature = text_proj(features_text)
image_feature = image_proj(features_image)

combined = torch.cat([text_feature, image_feature], dim=1)  # [1,256]
print("合并的特征向量为:")
print(combined)

##====== 分类器 ========##
class RumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
    def forward(self, combin):
        return self.classifier(combin)
    
#初始化组件
classifier0 = RumorClassifier()
loss = nn.CrossEntropyLoss()
adam = torch.optim.Adam(classifier0.parameters(), lr=0.001)  #优化

#模拟数据
examTest = torch.randn(4, 256)
tagTest = torch.tensor([0, 1, 0, 1])

print('开始训练begin:')
for i in range(10):
    classifier0.train()
    adam.zero_grad()
    
    proResult = classifier0(examTest)
    
    errorNow = loss(proResult, tagTest)
    
    errorNow.backward()
    adam.step()
    
    _, proAnswer = torch.max(proResult, 1)
    accuracy = (proAnswer == tagTest).float().mean()
    
    print(f"第{i+1}轮训练 | 错误值:{errorNow.item():.4f} | 正确率:{accuracy:.4f}")
    
with torch.no_grad():
    endPro = classifier0(combined)
    result = torch.argmax(endPro, dim=1)
    
print("\n==最终答案==")
print("谣言！"if result.item() == 0 else "真实")
print("特征向量实例：", combined[:1, :5])    