#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models

# -------------------------------
# 1. 定义数据集类：加载图像和 mask（不使用额外特征列）
# -------------------------------
class CPJUMP1DatasetAblation(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform_img=None, transform_mask=None):
        """
        Args:
            csv_file (str): metadata CSV 文件路径，需包含以下列：
                - FileName_OrigRNA: 图像文件名（包含扩展名）
                - Metadata_pert_iname: 化合物扰动名称
                - Metadata_target: 基因靶标
            img_dir (str): 图像文件夹路径（例如: ./downsampled_data/）
            mask_dir (str): mask 文件夹路径（文件名与图像一致）
            transform_img (callable, optional): 对图像的预处理
            transform_mask (callable, optional): 对 mask 的预处理
        """
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # 构建标签映射
        self.compound_names = sorted(self.metadata['Metadata_pert_iname'].unique())
        self.gene_targets = sorted(self.metadata['Metadata_target'].unique(), key=lambda x: str(x))
        self.compound_to_idx = {name: idx for idx, name in enumerate(self.compound_names)}
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_targets)}

        # 过滤掉图像文件或 mask 不存在的记录
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            file_prefix = row['FileName_OrigRNA'].split('_')[0]
            img_path = os.path.join(self.img_dir, file_prefix + '_median_aggregated.tiff')
            mask_path = os.path.join(self.mask_dir, 'MASK_' + file_prefix + '_median_aggregated.tif')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_indices.append(idx)
        if not valid_indices:
            raise ValueError("No valid image-mask pairs found in the dataset!")
        self.metadata = self.metadata.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_prefix = row['FileName_OrigRNA'].split('_')[0]

        # 加载聚合图像，文件名为 <file_prefix>_median_aggregated.tiff
        img_path = os.path.join(self.img_dir, file_prefix + '_median_aggregated.tiff')
        image = Image.open(img_path)
        if image.mode != 'L':
            image = image.convert('L')
        if self.transform_img:
            image = self.transform_img(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 加载对应的 segmentation mask，文件名为 MASK_<file_prefix>_median_aggregated.tif
        mask_path = os.path.join(self.mask_dir, 'MASK_' + file_prefix + '_median_aggregated.tif')
        mask = Image.open(mask_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = transforms.ToTensor()(mask)
        
        # 拼接为 2 通道输入
        x = torch.cat([image, mask], dim=0)  # shape: (2, H, W)
        
        # 获取分类标签
        compound_label = self.compound_to_idx[row['Metadata_pert_iname']]
        gene_label = self.gene_to_idx[row['Metadata_target']]

        return x, compound_label, gene_label


# -------------------------------
# 2. 定义模型：基于 EfficientNet-B0 提取图像特征建立多任务分类分支（不融合额外特征）
# -------------------------------
class MultiTaskClassifierEfficientNetAblation(nn.Module):
    def __init__(self, num_compound_classes, num_gene_classes, pretrained=True):
        super(MultiTaskClassifierEfficientNetAblation, self).__init__()
        # 加载预训练 EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        # 修改第一层卷积，使其支持 2 通道输入
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            2,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        with torch.no_grad():
            avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight = nn.Parameter(avg_weight.repeat(1, 2, 1, 1))
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias
        self.backbone.features[0][0] = new_conv
        
        # 移除 EfficientNet 的分类层，保留特征向量
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # 融合网络：直接对图像特征进行降维，再分别输出预测
        self.fuse = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 化合物分类分支
        self.compound_head = nn.Linear(256, num_compound_classes)
        # 基因目标分类分支
        self.gene_head = nn.Linear(256, num_gene_classes)
    
    def forward(self, x):
        img_features = self.backbone(x)  # shape: (B, in_features)
        fused = self.fuse(img_features)
        compound_out = self.compound_head(fused)
        gene_out = self.gene_head(fused)
        return compound_out, gene_out


# -------------------------------
# 3. 数据加载与训练集、验证集划分
# -------------------------------
if __name__ == "__main__":
    # 根据实际情况修改 CSV 和文件夹路径
    csv_file = '/data/user/fangxy/CU/drug_cell/merged_metadata_and_features.csv'
    img_dir = '/data/user/fangxy/CU/drug_cell/downsampled_data'
    mask_dir = '/data/user/fangxy/CU/drug_cell/downsampled_data/Masks'
    
    # 图像与 mask 预处理：对图像进行标准化处理
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 使用不含额外特征的消融数据集
    full_dataset = CPJUMP1DatasetAblation(csv_file, img_dir, mask_dir,
                                           transform_img=transform_img,
                                           transform_mask=transform_mask)
    
    # 划分训练集与验证集（验证集占 20%）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 获取类别数
    num_compound_classes = len(full_dataset.compound_to_idx)
    num_gene_classes = len(full_dataset.gene_to_idx)
    print(f'Compound classes: {num_compound_classes}, Gene target classes: {num_gene_classes}')
    
    # -------------------------------
    # 4. 模型、损失函数与优化器
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskClassifierEfficientNetAblation(num_compound_classes, num_gene_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # -------------------------------
    # 5. 训练与验证循环
    # -------------------------------
    num_epochs = 20
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, compound_labels, gene_labels in train_loader:
            inputs = inputs.to(device)  # shape: (B, 2, 256, 256)
            compound_labels = compound_labels.to(device)
            gene_labels = gene_labels.to(device)
            
            optimizer.zero_grad()
            compound_preds, gene_preds = model(inputs)
            loss_compound = criterion(compound_preds, compound_labels)
            loss_gene = criterion(gene_preds, gene_labels)
            loss = loss_compound + loss_gene
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / train_size
    
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_compound = 0
        correct_gene = 0
        total = 0
        with torch.no_grad():
            for inputs, compound_labels, gene_labels in val_loader:
                inputs = inputs.to(device)
                compound_labels = compound_labels.to(device)
                gene_labels = gene_labels.to(device)
                
                compound_preds, gene_preds = model(inputs)
                loss_compound = criterion(compound_preds, compound_labels)
                loss_gene = criterion(gene_preds, gene_labels)
                loss = loss_compound + loss_gene
                val_loss += loss.item() * inputs.size(0)
                
                _, compound_predicted = torch.max(compound_preds, 1)
                _, gene_predicted = torch.max(gene_preds, 1)
                total += inputs.size(0)
                correct_compound += (compound_predicted == compound_labels).sum().item()
                correct_gene += (gene_predicted == gene_labels).sum().item()
        
        val_loss /= val_size
        compound_acc = 100 * correct_compound / total
        gene_acc = 100 * correct_gene / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Compound Acc: {compound_acc:.2f}%, Gene Acc: {gene_acc:.2f}%")
