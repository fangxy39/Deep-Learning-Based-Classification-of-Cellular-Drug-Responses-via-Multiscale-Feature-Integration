#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MultiTask Classification for CPJUMP1 Dataset with EfficientNet-B0, Data Normalization, and Additional Features

该脚本实现了以下功能：
1. 从 CSV 文件中读取样本信息，并根据文件名前缀加载聚合图像和对应的 segmentation mask，同时提取额外的特征列。
2. 过滤掉图像或 mask 不存在的记录。
3. 构建一个基于预训练 EfficientNet-B0 的多任务分类模型（修改第一层卷积以支持 2 通道输入），并融合图像特征与额外特征，
   分别预测化合物扰动（Metadata_pert_iname）和基因靶标（Metadata_target）。
4. 对图像数据进行标准化处理（归一化到 [-1, 1] 或其它区间）。
5. 划分训练集与验证集，并进行训练和验证。

注意：请根据需要调整额外特征列的选择逻辑及训练超参数。
"""

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
# 1. 定义数据集类：加载图像、mask以及额外的特征数据
# -------------------------------
class CPJUMP1Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, transform_img=None, transform_mask=None):
        """
        Args:
            csv_file (str): metadata CSV 文件路径，需包含以下列：
                - FileName_OrigRNA: 图像文件名（包含扩展名）
                - Metadata_pert_iname: 化合物扰动名称
                - Metadata_target: 基因靶标
              以及额外的特征列（本例中，除上述3列外的所有列均视为额外特征）
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

        # 额外特征列：此处选取部分关键特征列
        self.feature_columns = [
            "Texture_DifferenceVariance_OrigRNA_10_03_256",
            'AreaShape_Area',               # cell area
            'AreaShape_Compactness',        # cell compactness
            'AreaShape_Eccentricity',       # cell eccentricity
            'AreaShape_Perimeter',          # cell perimeter
            'Intensity_MeanIntensity_OrigRNA',      # mean intensity
            'Intensity_IntegratedIntensity_OrigRNA',# integrated intensity
            'Granularity_5_OrigRNA',                # granularity measure
            'RadialDistribution_FracAtD_OrigRNA_4of4'   # radial distribution fraction
        ]

        # 过滤掉图像文件或 mask 不存在的记录
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            # 提取文件名前缀，例如 "r16c24f09p01-ch3sk1fk1fl1.tiff" 提取 "r16c24f09p01"
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

        # 加载单通道聚合图像，文件名为 <file_prefix>_median_aggregated.tiff
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
        
        # 额外特征：转换为浮点数 tensor
        features = row[self.feature_columns].values.astype(float)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # 获取分类标签
        compound_label = self.compound_to_idx[row['Metadata_pert_iname']]
        gene_label = self.gene_to_idx[row['Metadata_target']]

        return x, features_tensor, compound_label, gene_label


# -------------------------------
# 2. 定义模型：基于 EfficientNet-B0 提取图像特征，并融合额外特征建立多任务分类分支
# -------------------------------
class MultiTaskClassifierEfficientNet(nn.Module):
    def __init__(self, num_compound_classes, num_gene_classes, num_extracted_features, pretrained=True):
        super(MultiTaskClassifierEfficientNet, self).__init__()
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
        
        # 移除 EfficientNet 的分类层，提取特征向量
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # 小型网络用于处理额外特征，降维到 32
        self.feature_net = nn.Sequential(
            nn.Linear(num_extracted_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 融合图像特征与额外特征
        fused_dim = in_features + 32
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 化合物分类分支
        self.compound_head = nn.Linear(256, num_compound_classes)
        # 基因目标分类分支
        self.gene_head = nn.Linear(256, num_gene_classes)
    
    def forward(self, x, features):
        img_features = self.backbone(x)  # shape: (B, in_features)
        extra_features = self.feature_net(features)  # shape: (B, 32)
        fused_features = torch.cat([img_features, extra_features], dim=1)
        fused = self.fuse(fused_features)
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
    
    full_dataset = CPJUMP1Dataset(csv_file, img_dir, mask_dir,
                                  transform_img=transform_img,
                                  transform_mask=transform_mask)
    
    # 划分训练集和验证集（验证集占 20%）
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 获取类别数与额外特征维度
    num_compound_classes = len(full_dataset.compound_to_idx)
    num_gene_classes = len(full_dataset.gene_to_idx)
    num_extracted_features = len(full_dataset.feature_columns)
    print(f'Compound classes: {num_compound_classes}, Gene target classes: {num_gene_classes}')
    print(f'Number of extra features: {num_extracted_features}')
    
    # -------------------------------
    # 4. 模型、损失函数与优化器
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskClassifierEfficientNet(num_compound_classes, num_gene_classes, num_extracted_features, pretrained=True).to(device)
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
        for inputs, features, compound_labels, gene_labels in train_loader:
            inputs = inputs.to(device)
            features = features.to(device)
            compound_labels = compound_labels.to(device)
            gene_labels = gene_labels.to(device)
            
            optimizer.zero_grad()
            compound_preds, gene_preds = model(inputs, features)
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
            for inputs, features, compound_labels, gene_labels in val_loader:
                inputs = inputs.to(device)
                features = features.to(device)
                compound_labels = compound_labels.to(device)
                gene_labels = gene_labels.to(device)
                
                compound_preds, gene_preds = model(inputs, features)
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
