import torch

import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

from torchvision.models import mobilenet_v3_small

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

import os

# 🚀 轻量级特征提取器

class LiteFeatureExtractor:

    def __init__(self):

        # 微型文本模型

        self.text_model = AutoModel.from_pretrained("bert-base-chinese", output_hidden_states=True)

        self.text_proj = nn.Linear(768, 128)

        

        # 轻量图像模型

        self.image_model = mobilenet_v3_small(pretrained=True)

        self.image_model.classifier = nn.Identity()

        self.image_proj = nn.Linear(576, 128)

        

        # 图像预处理

        self.image_transform = transforms.Compose([

            transforms.Resize(224),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])

    def get_text_features(self, text_input):

        # 使用前3层隐藏状态的均值

        with torch.no_grad():

            outputs = self.text_model(**text_input)

        return self.text_proj(torch.stack(outputs.hidden_states[:3]).mean(0)[:,0])

    def get_image_features(self, image):

        with torch.no_grad():

            return self.image_proj(self.image_model(image))

# 🚀 高效数据集处理

class StreamlinedDataset(Dataset):

    def __init__(self, tsv_path, tokenizer, max_length=64, max_samples=20):

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.samples = self._parse_tsv(tsv_path)[:max_samples]

    def _parse_tsv(self, path):

        valid_samples = []

        with open(path, 'r', encoding='utf-8') as f:

            next(f)  # 跳过标题行

            for line in f:

                if len(line.split('\t')) >= 2:

                    parts = line.strip().split('\t')

                    img_tag = parts[1].split('[IMG:')[-1].split(']')[0]

                    sample = {

                        'text': parts[1].split('[IMG:')[0].strip(),

                        'label': int(parts[0]),

                        'image': os.path.join("data_test", img_tag)  # 直接使用data_test目录

                    }

                    valid_samples.append(sample)

        return valid_samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        inputs = self.tokenizer(

            item['text'],

            max_length=self.max_length,

            padding='max_length',

            truncation=True,

            return_tensors='pt'

        )

        return {

            'input_ids': inputs['input_ids'].squeeze(),

            'attention_mask': inputs['attention_mask'].squeeze(),

            'image_path': item['image'],

            'label': torch.tensor(item['label'], dtype=torch.long)

        }

# 🚀 紧凑型分类模型

class LiteRumorDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = LiteFeatureExtractor()

        self.classifier = nn.Sequential(

            nn.Linear(256, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(64, 2)

        )

    def forward(self, batch):

        # 文本特征提取

        text_feat = self.feature_extractor.get_text_features({

            'input_ids': batch['input_ids'],

            'attention_mask': batch['attention_mask']

        })

        

        # 增强的图像加载逻辑

        img_feat = []

        for img_path in batch['image_path']:

            try:

                if os.path.exists(img_path):

                    image = Image.open(img_path).convert('RGB')

                    tensor = self.feature_extractor.image_transform(image).unsqueeze(0)

                    img_feat.append(self.feature_extractor.get_image_features(tensor))

                else:

                    raise FileNotFoundError

            except Exception as e:

                # 生成设备兼容的零张量

                zero_feat = torch.zeros(1, 128, device=text_feat.device)

                img_feat.append(zero_feat)

                #print(f"空白特征替代缺失图片: {os.path.basename(img_path)}")

        

        img_feat = torch.stack(img_feat).squeeze(1)

        return self.classifier(torch.cat([text_feat, img_feat], dim=1))

# 🚀 优化训练器

class EfficientTrainer:

    def __init__(self, data_dir):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LiteRumorDetector().to(self.device)

        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=2e-4)

        self.criterion = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for batch in loader:

            # 修正设备转移逻辑

            batch = {

                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 

                for k, v in batch.items()

            }

            

            self.optimizer.zero_grad()

            

            with torch.cuda.amp.autocast():

                outputs = self.model(batch)

                loss = self.criterion(outputs, batch['label'])

            

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    

    

# 新增测试评估类

class ModelEvaluator:

    def __init__(self, model_path, data_dir):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LiteRumorDetector().to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.criterion = nn.CrossEntropyLoss()

    

    def evaluate(self, loader):

        self.model.eval()

        total_loss = 0

        correct = 0

        all_preds = []

        all_labels = []

        error_samples = []

        

        with torch.no_grad():

            for batch_idx, batch in enumerate(loader):

                # 设备转移

                batch = {

                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v 

                    for k, v in batch.items()

                }

                

                outputs = self.model(batch)

                loss = self.criterion(outputs, batch['label'])

                total_loss += loss.item()

                

                # 计算准确率

                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == batch['label']).sum().item()

                all_preds.extend(predicted.cpu().numpy())

                all_labels.extend(batch['label'].cpu().numpy())

                

                # 收集错误样本

                wrong_mask = predicted != batch['label']

                for i in torch.where(wrong_mask)[0]:

                    idx = batch_idx * loader.batch_size + i.item()

                    error_samples.append({

                        'text': loader.dataset.samples[idx]['text'],

                        'true_label': batch['label'][i].item(),

                        'pred_label': predicted[i].item()

                    })

        

        accuracy = correct / len(loader.dataset)

        return {

            'loss': total_loss / len(loader),

            'accuracy': accuracy,

            'error_samples': error_samples[:5]  # 显示前5个错误样本

        }

if __name__ == "__main__":

    # 初始化组件

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    train_set = StreamlinedDataset(

        'D:/文档/项目-大二/create_pytorch/data_test/train.tsv',

        tokenizer,

        max_samples=1000

    )

    

    # 验证数据集

    print(f"样本示例：{train_set[0]['image_path']}")

    

    train_loader = DataLoader(

        train_set,

        batch_size=4,

        shuffle=True,

        num_workers=0  # 调试时设为0显示完整错误

    )

    # 训练模型

    trainer = EfficientTrainer('./data')

    print(f"开始训练，总样本量: {len(train_set)}")

    

    for epoch in range(5):

        loss = trainer.train_epoch(train_loader)

        print(f"Epoch {epoch+1} | 平均损失: {loss:.4f}")

    # 保存模型

    torch.save(trainer.model.state_dict(), 'lite_rumor_detector.pth')

    print("训练完成，模型已保存")

    

     # 新增评估部分

    print("\n开始评估测试集...")

    

    # 加载测试集

    test_set = StreamlinedDataset(

        'D:/文档/项目-大二/create_pytorch/data_test/dev.tsv',  # 修改为实际路径

        tokenizer,

        max_samples=10

    )

    test_loader = DataLoader(

        test_set,

        batch_size=4,

        shuffle=False,

        num_workers=0

    )

    

    # 初始化评估器

    evaluator = ModelEvaluator('lite_rumor_detector.pth', './data')

    results = evaluator.evaluate(test_loader)

    

    # 打印评估结果

    print(f"\n评估结果：")

    print(f"平均损失: {results['loss']:.4f}")

    print(f"准确率: {results['accuracy']:.2%}")

    print("\n典型错误案例分析：")

    for i, sample in enumerate(results['error_samples']):

        print(f"案例 {i+1}:")

        print(f"真实标签: {'谣言' if sample['true_label'] else '非谣言'}")

        print(f"预测标签: {'谣言' if sample['pred_label'] else '非谣言'}")

        print(f"文本内容: {sample['text'][:50]}...\n")

    

    # 合理性分析

    print("合理性评估：")

    if results['accuracy'] > 0.7:

        print("模型表现出较好的分类能力，预测结果与人工判断基本一致")

    else:

        print("模型性能有待提升，建议检查以下方面：")

        print("- 图片特征提取的有效性")

        print("- 文本截断是否丢失关键信息")

        print("- 类别样本是否均衡")

    

    print("\n图片处理统计：")

    missing_count = sum(1 for s in test_set.samples if not os.path.exists(s['image']))

    print(f"缺失图片数量: {missing_count}/{len(test_set)}")