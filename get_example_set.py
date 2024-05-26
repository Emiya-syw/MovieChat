import json
import torch
import torch.nn as nn
import argparse
import os 
import clip
import tqdm
import math

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster image_embeds with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            image_embeds are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape
        
        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)    # 计算两组点之间的距离, 返回形状为(B, N, N)的张量
        
        if token_mask is not None:
            token_mask = token_mask > 0
            # 为了不影响局部密度, 空的token和其他token之间的距离应该是最大值
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                (dist_matrix.max() + 1) * (~token_mask[:, None, :])
        
        # 计算局部密度 
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # 每个点的密度  (B, N)
        # 增加一点噪声, 确保token之间不享有相同的密度
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6
        
        if token_mask is not None:
            # 空token的密度应该是0
            density = density * token_mask
        
        # 得到距离指数
        mask = density[:, None, :] > density[:, :, None]    # 增加维度 (B, 1, N) (B, N, 1)  -> (B, N, N)
        mask = mask.type(x.dtype)  
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] # 对每个点的距离最大值 -> (B, 1, 1)
        dist, index_parent = (dist_matrix * mask + dist_max * (1-mask)).min(dim=-1)
        
        # 根据得分选择聚类中心
        score = dist * density  # 指数 * 密度
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)    # 每个样本中score较高的k个点的索引
        
        # 将token分配到最近的聚类中心
        dist_matrix = index_points(dist_matrix, index_down) # 返回每个样本中聚类中心的距离向量
        idx_cluster = dist_matrix.argmin(dim=1) # 中心本身
        
        # 确保聚类中心合并到它本身
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)    # (B, K)
        # print(idx_batch.shape)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        # print(idx_tmp.shape)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        # print(idx_cluster.shape)
        
    
    # 聚类后的索引分布, 聚类数量
    return idx_cluster, cluster_num, index_down

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge image_embeds in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]   # (B, 1)
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight

    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = None
    return out_dict

class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.k = k

    def forward(self, token_dict, sample_ratio=None):
        x = token_dict["x"]
        B, N, C = x.shape

        token_weight = x.new_ones(B, N)

        if token_dict["mask"] is not None:
            token_weight.masked_fill_((1 - token_dict["mask"]).to(torch.bool), float("-inf"))
        token_weight = token_weight.unsqueeze(2)
        token_dict['x'] = x

        if sample_ratio is not None:
            cluster_num = max(math.ceil(N * sample_ratio), 1)
        elif self.sample_ratio > 1:
            cluster_num = max(math.ceil(self.sample_ratio), 1)
        else:
            cluster_num = max(math.ceil(N * self.sample_ratio), 1)

        k = min(3, max(cluster_num//2, 1)) if self.k > cluster_num else self.k
        idx_cluster, cluster_num, center_id = cluster_dpc_knn(
            token_dict, cluster_num, k, token_mask=token_dict["mask"])

        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        return down_dict, token_dict, center_id

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="global")
    args.add_argument("--json-path", type=str, default="./jsons")
    args.add_argument("--output", type=str, default="./Outputs/examples.json")
    args.add_argument("--sample-ratio", type=float, default=0.05)
    
    return args.parse_args()
    
    
def main(args):
    mode = args.mode
    path = args.json_path
    device = "cuda:0"
    bs = 64
    qa_list = []
    questions_tokenizer = []
    questions_features = []
    examples = {}
    for filename in tqdm.tqdm(os.listdir(path)):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as f:
            sample = json.load(f)
            for qa in sample[mode]:
                qa_list.append((qa["question"], qa["answer"]))
                questions_tokenizer.append(clip.tokenize(qa["question"]))
    model, _ = clip.load("./ckpt/ViT-B-32.pt", device=device)
    with torch.no_grad():
        for i in range(0, len(questions_tokenizer), 64):
            if i + 64 < len(questions_tokenizer):
                texts = torch.cat(questions_tokenizer[i:i+64]).to(device)
            else:
                texts = torch.cat(questions_tokenizer[i:]).to(device)
            questions_features.append(model.encode_text(texts))
        questions_features = torch.cat(questions_features, dim=0).unsqueeze(0).to(device)
        print(questions_features.shape)
    
    cluster_maker = CTM(args.sample_ratio, 512, 512)
    questions_features = {"x": questions_features,
                  "token_num": questions_features.size(1),  # 每个样本的token数量
                  "idx_token": torch.arange(questions_features.size(1))[None, :].repeat(questions_features.size(0), 1), # token的索引
                  "agg_weight": questions_features.new_ones(questions_features.size(0), questions_features.size(1), 1), # token聚合的权重, 全是1
                  "mask": None}
    _, _, center_id = cluster_maker(questions_features)
    for id in list(center_id.squeeze(0)):
        examples[qa_list[id][0]] = qa_list[id][1]
    with open(args.output, 'w') as f:
        json.dump(examples, f)
    
    
    

if __name__ == "__main__":
    args = get_args()
    main(args)

