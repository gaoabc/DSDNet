import torch
import torch.nn as nn
import torch.nn.functional as F
# def calculate_class_centroids(predictions, labels, num_classes):
#     """
#     计算每个类别的类别中心
#     predictions: Tensor of shape [batch_size, num_channels, height, width]
#     labels: Tensor of shape [batch_size, height, width]
#     num_classes: 类别数量
#     """
#     batch_size, num_channels, height, width = predictions.size()
#     centroids = torch.zeros(num_classes, num_channels, device=predictions.device)  # [num_classes, num_channels]
#     count = torch.zeros(num_classes, device=predictions.device)  # [num_classes]
#
#     for b in range(batch_size):  # 遍历batch
#         for h in range(height):  # 遍历高度
#             for w in range(width):  # 遍历宽度
#                 label = labels[b, h, w]  # 获取该位置的真实标签
#                 if label == 0:  # 排除背景
#                     continue
#
#                 # 累加像素特征到相应类别中心
#                 centroids[label] += predictions[b, :, h, w]
#                 count[label] += 1
#
#     # 通过像素数来归一化类别中心
#     for k in range(1, num_classes):  # 排除背景类
#         if count[k] > 0:
#             centroids[k] /= count[k]
#
#     return centroids
#
# def cosine_similarity(x, y):
#     """
#     计算两个向量的余弦相似度
#     x: [batch_size, C]
#     y: [batch_size, C]
#     """
#     return F.cosine_similarity(x, y, dim=1)
#
#
# def contrastive_loss(predictions, centroids, labels, num_classes, temperature=0.1):
#     loss = 0.0
#     batch_size, num_channels, height, width = predictions.size()
#
#     for b in range(batch_size):  # 遍历batch
#         for h in range(height):  # 遍历高度
#             for w in range(width):  # 遍历宽度
#                 label = labels[b, h, w]  # 获取该位置的真实标签
#                 if label == 0:  # 如果是背景，跳过
#                     continue
#
#                 pixel_feature = predictions[b, :, h, w]  # 获取该像素的特征
#                 centroid = centroids[label]  # 获取该类别的类别中心
#
#                 # 正样本对：像素特征与该类别的类别中心
#                 positive_similarity = cosine_similarity(pixel_feature.unsqueeze(0), centroid.unsqueeze(0))
#
#                 # 负样本对：计算当前像素特征与背景类别的类别中心的相似度
#                 negative_similarities = []
#                 negative_label = 0 if label == 1 else 1  # 如果是前景，负样本为背景，反之亦然
#                 negative_similarity = cosine_similarity(pixel_feature.unsqueeze(0),
#                                                         centroids[negative_label].unsqueeze(0))
#                 negative_similarities.append(negative_similarity)
#
#                 # 将所有相似度（正负样本）拼接成一个一维张量
#                 all_similarities = torch.cat([positive_similarity.unsqueeze(0), torch.stack(negative_similarities)],
#                                              dim=0)
#                 labels_sim = torch.zeros(all_similarities.size(0))  # 正样本标签为0，负样本标签为1
#                 labels_sim[0] = 1  # 正样本标签为1
#
#                 # 计算对比损失
#                 exp_sim = torch.exp(all_similarities / temperature)
#                 contrastive_loss = -torch.log(exp_sim[0] / exp_sim.sum())
#                 loss += contrastive_loss
#
#     return loss / (batch_size * height * width)
def calculate_class_centroids(predictions, labels, num_classes):
    """
    矢量化类别中心计算
    predictions: Tensor [batch_size, num_channels, height, width]
    labels: Tensor [batch_size, height, width]
    num_classes: 类别数量
    """
    batch_size, num_channels, height, width = predictions.size()

    # 扁平化预测和标签
    predictions = predictions.view(batch_size, num_channels, -1)  # [batch_size, num_channels, height*width]
    labels = labels.view(batch_size, -1)  # [batch_size, height*width]

    # 初始化类别中心和计数
    centroids = torch.zeros(num_classes, num_channels, device=predictions.device)
    counts = torch.zeros(num_classes, device=predictions.device)

    # 遍历每个类别，计算类别中心
    for c in range(1, num_classes):  # 跳过背景类
        # 获取当前类别的掩码
        mask = (labels == c).float()  # [batch_size, height*width]

        # 计算每个类别的特征和数量
        masked_features = predictions * mask.unsqueeze(1)  # 广播到 [batch_size, num_channels, height*width]
        centroids[c] = masked_features.sum(dim=(0, 2))
        counts[c] = mask.sum()

        # 归一化
        if counts[c] > 0:
            centroids[c] /= counts[c]

    return centroids


def contrastive_loss(predictions, centroids, labels, temperature=0.1):
    """
    批量化对比损失计算
    predictions: [batch_size, num_channels, height, width]
    centroids: [num_classes, num_channels]
    labels: [batch_size, height, width]
    """
    batch_size, num_channels, height, width = predictions.size()

    # 扁平化处理
    predictions = predictions.view(batch_size, num_channels, -1)  # [batch_size, num_channels, height*width]
    labels = labels.view(batch_size, -1)  # [batch_size, height*width]

    # 正样本和负样本处理
    mask_positive = (labels == 1).float()  # 前景掩码
    mask_negative = (labels == 0).float()  # 背景掩码

    # 提取正负样本特征
    positive_features = predictions * mask_positive.unsqueeze(1)
    negative_features = predictions * mask_negative.unsqueeze(1)

    # 计算余弦相似度
    pos_sim = F.cosine_similarity(positive_features.permute(0, 2, 1), centroids[1].unsqueeze(0), dim=2)
    neg_sim = F.cosine_similarity(negative_features.permute(0, 2, 1), centroids[0].unsqueeze(0), dim=2)

    # 对比损失计算
    exp_pos = torch.exp(pos_sim / temperature)
    exp_neg = torch.exp(neg_sim / temperature)

    # 避免数值问题，计算对比损失
    contrastive_loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-7))

    # 仅计算正样本损失
    loss = contrastive_loss * mask_positive
    return loss.sum() / mask_positive.sum()




