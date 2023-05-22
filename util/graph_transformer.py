import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, max_length):
        super(GraphTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self._generate_positional_encoding(hidden_dim, max_length)

        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.compress = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_embedding):
        embedded_graph = self.embedding(graph_embedding)

        # 添加位置编码
        positional_encoding = self.positional_encoding[:graph_embedding.shape[0], :]
        embedded_graph += positional_encoding

        # Transformer encoding
        encoded_graph = self.transformer_encoder(embedded_graph)

        # Data compression
        compressed_graph = self.compress(encoded_graph)

        return compressed_graph

    def _generate_positional_encoding(self, d_model, max_length):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(max_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding


def encoding(embeddings):
    embedding_matrix = torch.tensor(embeddings.vectors, dtype=torch.float32)

    # 使用示例
    input_dim = 64
    hidden_dim = 256
    output_dim = 32
    num_layers = 4
    num_heads = 8
    dropout = 0.2
    max_length = 10

    # 创建模型
    model = GraphTransformer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, max_length)

    # 前向传播
    output = model(embedding_matrix)
    return output
