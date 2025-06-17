import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RVGAE(nn.Module):
    """
    Relational Variational Graph Autoencoder (RVGAE)

    이 모델은 R-GCN(Relational Graph Convolutional Network)을 기반으로 한 변분 그래프 오토인코더입니다.
    노드 임베딩을 통해 관계 예측 및 관계 유형 분류를 동시에 수행합니다.

    Args:
        in_channels (int): 입력 피처 차원
        hidden_channels (int): 은닉층 차원
        out_channels (int): 잠재 임베딩(latent) 차원
        num_relations (int): 관계 유형의 개수 (edge type 개수)

    Architecture:
        - Encoder: 3-layer RGCN (Base → Mean/LogStd 분기)
        - Decoder: 공유 MLP 기반 분기형 디코더
            - Link Prediction: 이진 분류 (링크 존재 여부)
            - Type Prediction: 다중 분류 (링크 관계 유형)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RVGAE, self).__init__()
        self.num_relations = num_relations

        # === 인코더 (R-GCN 기반) ===
        self.rgcn_base = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn_mean = RGCNConv(hidden_channels, out_channels, num_relations)
        self.rgcn_logstd = RGCNConv(hidden_channels, out_channels, num_relations)

        # === 공유 디코더 ===
        self.shared_decoder = nn.Sequential(
            nn.Linear(2 * out_channels, 128),
            nn.ReLU()
        )

        # === 출력층 분기 ===
        self.link_out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.type_out = nn.Linear(128, num_relations)  # 다중 클래스 분류

    def encode(self, x, edge_index, edge_type):
        """
        인코더: 입력 노드 피처를 잠재 벡터(z)로 인코딩

        Args:
            x (Tensor): 노드 피처 텐서 (num_nodes, in_channels)
            edge_index (Tensor): 엣지 인덱스 (2, num_edges)
            edge_type (Tensor): 각 엣지의 관계 타입 (num_edges,)

        Returns:
            z (Tensor): 샘플링된 잠재 벡터 (num_nodes, out_channels)
            mean (Tensor): 정규분포 평균 (num_nodes, out_channels)
            logstd (Tensor): 정규분포 로그표준편차 (num_nodes, out_channels)
        """
        h = F.relu(self.rgcn_base(x, edge_index, edge_type))
        mean = self.rgcn_mean(h, edge_index, edge_type)
        logstd = self.rgcn_logstd(h, edge_index, edge_type)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logstd

    def decode(self, z, edge_index):
        """
        디코더: 노드 임베딩을 입력으로 링크 존재 여부와 관계 유형을 예측

        Args:
            z (Tensor): 노드 임베딩 (num_nodes, out_channels)
            edge_index (Tensor): 예측 대상 엣지 인덱스 (2, num_pred_edges)

        Returns:
            link_pred (Tensor): 링크 존재 확률 (num_pred_edges,)
            type_pred (Tensor): 관계 유형 logits (num_pred_edges, num_relations)
        """
        src = z[edge_index[0]] # => edge_index[0]는 (num_edges,)짜리 텐서
        dst = z[edge_index[1]] # => [edge_index[0]]는 (num_edges, embedding_dim)
        z_pair = torch.cat([src, dst], dim=1)  # shape: (batch_size, 2*out_dim)

        share = self.shared_decoder(z_pair)
        link_pred = self.link_out(share).squeeze()  # shape: (batch_size,)
        type_pred = self.type_out(share)          # shape: (batch_size, num_relations)

        return link_pred, type_pred

    def forward(self, x, edge_index, edge_type, pos_edge_index):
        """
        전체 순전파 (인코더 + 디코더)

        Args:
            x (Tensor): 노드 피처 (num_nodes, in_channels)
            edge_index (Tensor): 훈련용 그래프 엣지 인덱스 (2, num_edges)
            edge_type (Tensor): 훈련용 엣지 관계 타입 (num_edges,)
            pos_edge_index (Tensor): 예측 대상 엣지 인덱스 (2, num_pred_edges)

        Returns:
            link_pred (Tensor): 예측 링크 존재 확률
            type_pred (Tensor): 예측 관계 유형 logits
            mean (Tensor): 정규분포 평균 (잠재 분포)
            logstd (Tensor): 정규분포 로그표준편차
            z (Tensor): 샘플링된 잠재 임베딩
        """
        z, mean, logstd = self.encode(x, edge_index, edge_type)
        link_pred, type_pred = self.decode(z, pos_edge_index)
        return link_pred, type_pred, mean, logstd, z