import torch


def diag_scan(x, H, W):
    B, K, _, L = x.shape
    assert L == H * W, "最后一个维度L必须等于H*W"

    # 创建二维的索引矩阵
    idx = torch.arange(H * W, device=x.device).reshape(H, W)

    # 生成对角线索引
    i_idx = torch.arange(H, device=x.device).reshape(-1, 1).expand(H, W)  # 行索引 (H, 1) => (H, W)
    j_idx = torch.arange(W, device=x.device).reshape(1, -1).expand(H, W)  # 列索引 (1, W) => (H, W)
    diag_mask = (i_idx + j_idx)

    sorted_idx = torch.argsort(diag_mask.view(-1))
    diag_indices = torch.index_select(idx.view(-1), 0, sorted_idx)

    # 展平输入
    x_flat = x.view(B, K, -1, H * W)

    # 按对角线顺序重新排列
    diag_flat = torch.index_select(x_flat, dim=3, index=diag_indices)

    return diag_flat, diag_indices


def reverse_diag_scan(x, diag_indices):
    B, K, _, L = x.shape

    # 展平输入
    x_flat = x.view(B, K, -1, L)

    # 创建反向索引
    reverse_indices = torch.argsort(diag_indices)

    # 恢复原始顺序
    original_flat = torch.index_select(x_flat, dim=3, index=reverse_indices)

    return original_flat


# 示例用法
B, K, _, H, W = 2, 2, 1, 3, 3
x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
x = x.repeat(B, K, 1, 1, 1)  # 制造一个(B=2, K=2, 1, H=3, W=3)的示例张量进行展示
x = x.view(B, K, -1, H * W)  # 转换为(B, K, 1, H*W)

diag_repr, diag_indices = diag_scan(x, H, W)
print("对角线扫描结果展平: \n", diag_repr)
print("对角线索引: \n", diag_indices)

original_repr = reverse_diag_scan(diag_repr, diag_indices)
print("恢复到原始形状展平: \n", original_repr)