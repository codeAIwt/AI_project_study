#模型分词器加载
from transformers import AutoModel,AutoTokenizer
#加载模型
model_name = r'\bert-data-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model0 = AutoModel.from_pretrained(model_name)

#文本处理
text_path = r'C:\Users\文档\Desktop\torchCS.txt'

with open(text_path, 'r',encoding='utf-8') as f:#打开文本
    text = f.read()#读取文本

inputs = tokenizer(text, return_tensors="pt", padding = True, truncation = True)#文本预处理

#图片处理
from PIL import Image
from torchvision import models, transforms

img = Inmage.open(r"C:\Users\文档\Desktop\mei.jpg")#打开图片

preprocess = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(224),#图像的中心裁剪出一个224x224的区域
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],#标准化（ImageNet统计值）(RGB通道的均值)
        std=[0.229, 0.224, 0.225]#标准化（ImageNet统计值）(RGB通道的标准差)
    )
])

input_tensor = preprocess(img)#使用preprocess对图片预处理

