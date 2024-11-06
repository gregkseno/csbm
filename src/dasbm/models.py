from typing import List, Tuple

import torch
from torch import nn

from dasbm.data import Prior, MoleculeBatch

def masked_softmax(x: torch.Tensor, mask: torch.Tensor, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

def apply_mask(
    nodes: torch.Tensor, 
    edges: torch.Tensor,
    mask: torch.Tensor, 
    collapse: bool = False
):
    nodes_mask = mask.unsqueeze(-1) # bs, n, 1
    edges_mask1 = nodes_mask.unsqueeze(2) # bs, n, 1, 1
    edges_mask2 = nodes_mask.unsqueeze(1) # bs, 1, n, 1

    if collapse:
        nodes = torch.argmax(nodes, dim=-1)
        edges = torch.argmax(edges, dim=-1)

        nodes[mask == 0] = -1
        edges[(edges_mask1 * edges_mask2).squeeze(-1) == 0] = -1
    else:
        nodes = nodes * nodes_mask
        edges = edges * edges_mask1 * edges_mask2
        assert torch.allclose(edges, torch.transpose(edges, 1, 2))
    return nodes, edges

class DiffusionModel(nn.Module):
    def __init__(
         self,
         input_dim: int,
         num_categories: int, 
         num_timesteps: int,
         timestep_dim: int = 2, 
         layers: List[int] = [128, 128, 128],
     ) -> None: 
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps
        net = []
        ch_prev = input_dim + timestep_dim
        for ch_next in layers:
            net.extend([nn.Linear(ch_prev, ch_next), nn.ReLU()])
            ch_prev = ch_next
        net.append(nn.Linear(ch_prev, num_categories * input_dim))
        self.net = nn.Sequential(*net)
        self.timestep_embedding = nn.Embedding(num_timesteps + 2, timestep_dim)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_start_logits = self.net(torch.cat([x_t, self.timestep_embedding(t)], dim=1))
        return x_start_logits.reshape(-1, self.input_dim, self.num_categories)
         

class D3PM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        prior: Prior,
    ) -> None:
        super().__init__()
        self.model = model
        self.prior = prior
        self.num_categories = model.num_categories
        self.num_timesteps = model.num_timesteps
        self.eps = 1e-6


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def markov_sample(self, x: torch.Tensor, t: torch.Tensor):
        """Samples from $p(x_{t-1} | x_{t}, \hat{x_{0}})$, where $\hat{x_{0}} \sim m_{\theta}(\hat{x_{0}} | x_{t})$.""" # type: ignore
        # noise = torch.rand((*x.shape, self.num_categories), device=self.device)
        # noise = torch.clip(noise, self.eps, 1.0)
        # gumbel_noise = -torch.log(-torch.log(noise))
        # not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        batch_size, dim = x.shape[:2]

        pred_x_end_logits = self(x, t)
        pred_q_posterior_logits = self.prior.q_posterior_logits(pred_x_end_logits, x, t)
        probs = pred_q_posterior_logits.softmax(-1)
        probs = probs.reshape(batch_size * dim, -1)
        # remove gumbel noise at `t==1` because we already have correct logits
        # torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)
        return probs.multinomial(num_samples=1).reshape(x.shape)
        
    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t)
        return x
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        trajectory = [x]
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory
    
    @property
    def device(self):
        return next(self.parameters()).device


class Nodes2Features(nn.Module):
    def __init__(self, input_dim: int, features_dim: int):
        super().__init__()
        self.net = nn.Linear(4 * input_dim, features_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        z = torch.hstack((
            nodes.mean(dim=1), 
            nodes.min(dim=1)[0], 
            nodes.max(dim=1)[0], 
            nodes.std(dim=1)
        ))
        out = self.net(z)
        return out


class Edges2Features(nn.Module):
    def __init__(self, input_dim: int, features_dim: int):
        super().__init__()
        self.net = nn.Linear(4 * input_dim, features_dim)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        z = torch.hstack((
            edges.mean(dim=(1, 2)), 
            edges.min(dim=2)[0].min(dim=1)[0], 
            edges.max(dim=2)[0].max(dim=1)[0], 
            edges.std(dim=(1, 2))
        ))
        return self.net(z)

class FiLMSelfAttention(nn.Module):
    def __init__(
        self, 
        nodes_dim: int, 
        edges_dim: int, 
        features_dim: int, 
        num_heads: int,
    ):
        super().__init__()
        assert nodes_dim % num_heads == 0, f"dx: {nodes_dim} -- nhead: {num_heads}"
        self.df = int(nodes_dim / num_heads)
        self.num_heads = num_heads

        # Attention
        self.q = nn.Linear(nodes_dim, nodes_dim)
        self.k = nn.Linear(nodes_dim, nodes_dim)
        self.v = nn.Linear(nodes_dim, nodes_dim)

        # FiLM edges to nodes
        self.edges_nodes_add = nn.Linear(edges_dim, nodes_dim)
        self.edges_nodes_mul = nn.Linear(edges_dim, nodes_dim)

        # FiLM features to edges
        self.features_edges_mul = nn.Linear(features_dim, nodes_dim)     
        self.features_edges_add = nn.Linear(features_dim, nodes_dim)

        # FiLM y to X
        self.features_nodes_mul = nn.Linear(features_dim, nodes_dim)
        self.features_nodes_add = nn.Linear(features_dim, nodes_dim)

        # Process y
        self.features_to_features = nn.Linear(features_dim, features_dim)
        self.nodes_to_features = Nodes2Features(nodes_dim, features_dim)
        self.edges_to_features = Edges2Features(edges_dim, features_dim)

        # Output layers
        self.nodes_net = nn.Linear(nodes_dim, nodes_dim)
        self.edges_net = nn.Linear(nodes_dim, edges_dim)
        self.features_net = nn.Sequential(
            nn.Linear(features_dim, features_dim), 
            nn.ReLU(), 
            nn.Linear(features_dim, features_dim)
        )

    def forward(
        self, 
        nodes: torch.Tensor, 
        edges: torch.Tensor, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = nodes.shape[1]
        nodes_mask = mask.unsqueeze(-1) # bs, n, 1
        edges_mask1 = nodes_mask.unsqueeze(2) # bs, n, 1, 1
        edges_mask2 = nodes_mask.unsqueeze(1) # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(nodes) * nodes_mask
        K = self.k(nodes) * nodes_mask
        assert (Q * (1 - nodes_mask.long())).abs().max().item() < 1e-4, 'Variables not masked properly.'
        
        # 2. Reshape to (bs, n, num_heads, df) with dx = num_heads * df
        Q = Q.reshape((Q.shape[0], Q.shape[1], self.num_heads, self.df))
        K = K.reshape((K.shape[0], K.shape[1], self.num_heads, self.df))

        Q = Q.unsqueeze(2) # (bs, 1, n, num_heads, df)
        K = K.unsqueeze(1) # (bs, n, 1, num_heads, df)

        # Compute unnormalized attentions
        QK = Q * K # (bs, n, n, num_heads, df)
        QK = QK / (QK.shape[-1] ** 0.5)
        assert (QK * (1 - (edges_mask1 * edges_mask2).unsqueeze(-1).long())).abs().max().item() < 1e-4, 'Variables not masked properly.'

        edges1 = self.edges_nodes_mul(edges) * edges_mask1 * edges_mask2 # bs, n, n, nodes_dim
        edges1 = edges1.reshape((edges.shape[0], edges.shape[1], edges.shape[2], self.num_heads, self.df))

        edges2 = self.edges_nodes_add(edges) * edges_mask1 * edges_mask2 # bs, n, n, nodes_dim
        edges2 = edges2.reshape((edges.shape[0], edges.shape[1], edges.shape[2], self.num_heads, self.df))

        # Incorporate edge features to the self attention scores.
        QK = QK * (edges1 + 1) + edges2 # (bs, n, n, num_heads, df)

        # Incorporate `features` to `edges`
        new_edges = QK.flatten(start_dim=3) # bs, n, n, nodes_dim
        features_edges1 = self.features_edges_add(features).unsqueeze(1).unsqueeze(1) # bs, 1, 1, edges_dim
        features_edges2 = self.features_edges_mul(features).unsqueeze(1).unsqueeze(1)
        new_edges = features_edges1 + (features_edges2 + 1) * new_edges

        # Output `edges`
        new_edges = self.edges_net(new_edges) * edges_mask1 * edges_mask2 # bs, n, n, edges_dim
        assert (new_edges * (1 - (edges_mask1 * edges_mask2).long())).abs().max().item() < 1e-4, 'Variables not masked properly.'

        # Compute attentions
        softmax_mask = edges_mask2.expand(-1, n, -1, self.num_heads) # bs, 1, n, 1
        attention = masked_softmax(QK, softmax_mask, dim=2) # bs, n, n, n_head

        V = self.v(nodes) * nodes_mask # bs, n, nodes_dim
        V = V.reshape((V.shape[0], V.shape[1], self.num_heads, self.df))
        V = V.unsqueeze(1) # (bs, 1, n, num_heads, df)

        # Compute weighted values
        weighted_V = attention * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2) # bs, n, nodes_dim

        # Incorporate `features` to `nodes`
        features_nodes1 = self.features_nodes_add(features).unsqueeze(1)
        features_nodes2 = self.features_nodes_mul(features).unsqueeze(1)
        new_nodes = features_nodes1 + (features_nodes2 + 1) * weighted_V

        # Output `nodes`
        new_nodes = self.nodes_net(new_nodes) * nodes_mask
        assert (new_nodes * (1 - (nodes_mask).long())).abs().max().item() < 1e-4, 'Variables not masked properly.'

        # Process `features` based on `nodes` and `edges`
        features = self.features_to_features(features)
        edges_features = self.edges_to_features(edges)
        nodes_features = self.nodes_to_features(nodes)
        new_features = features + nodes_features + edges_features
        new_features = self.features_net(new_features)               # bs, dy

        return new_nodes, new_edges, new_features

class TransformerLayer(nn.Module):
    def __init__(
            self, 
            nodes_input_dim: int, 
            edges_input_dim: int, 
            features_input_dim: int, 
            num_heads: int, 
            nodes_hidden_dim: int = 2048,
            edges_hidden_dim: int = 128, 
            features_hidden_dim: int = 2048, 
            dropout_prob: float = 0.1,
            layer_norm_eps: float = 1e-5, 
        ) -> None:
        super().__init__()

        self.self_attention = FiLMSelfAttention(nodes_input_dim, edges_input_dim, features_input_dim, num_heads)
        
        
        self.nodes_dropout = nn.Dropout(dropout_prob)
        self.nodes_norm1 = nn.LayerNorm(nodes_input_dim, eps=layer_norm_eps)
        self.nodes_mlp = nn.Sequential(
            nn.Linear(nodes_input_dim, nodes_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nodes_hidden_dim, nodes_input_dim),
            nn.Dropout(dropout_prob)
        )
        self.nodes_norm2 = nn.LayerNorm(nodes_input_dim, eps=layer_norm_eps)

        self.edges_dropout = nn.Dropout(dropout_prob)
        self.edges_norm1 = nn.LayerNorm(edges_input_dim, eps=layer_norm_eps)
        self.edges_mlp = nn.Sequential(
            nn.Linear(edges_input_dim, edges_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(edges_hidden_dim, edges_input_dim),
            nn.Dropout(dropout_prob)
        )
        self.edges_norm2 = nn.LayerNorm(edges_input_dim, eps=layer_norm_eps)

        self.features_dropout = nn.Dropout(dropout_prob)
        self.features_norm1 = nn.LayerNorm(features_input_dim, eps=layer_norm_eps)
        self.features_mlp = nn.Sequential(
            nn.Linear(features_input_dim, features_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(features_hidden_dim, features_input_dim),
            nn.Dropout(dropout_prob)
        )
        self.features_norm2 = nn.LayerNorm(features_input_dim, eps=layer_norm_eps)

    def forward(
        self, 
        nodes: torch.Tensor, 
        edges: torch.Tensor, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_nodes, new_edges, new_features = self.self_attention(
            nodes, edges, features, 
            mask=mask
        )
        # Add + Normalize
        new_nodes = self.nodes_norm1(nodes + self.nodes_dropout(new_nodes))
        new_edges = self.edges_norm1(edges + self.edges_dropout(new_edges))
        new_features = self.features_norm1(features + self.features_dropout(new_features))

        # MLP + Add + Normalize
        new_nodes = self.nodes_norm2(new_nodes + self.nodes_mlp(new_nodes))
        new_edges = self.edges_norm2(new_edges + self.edges_mlp(new_edges))
        new_features= self.features_norm2(new_features + self.features_mlp(new_features))

        return new_nodes, new_edges, new_features

class GraphTransformer(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        nodes_input_dim: int,
        edges_input_dim: int, 
        features_input_dim: int, 
        nodes_hidden_dim: int = 256,
        edges_hidden_dim: int = 128,
        features_hidden_dim: int = 128, 
        num_heads: int = 8,
        nodes_tranforemer_input_dim: int = 256,
        edges_tranforemer_input_dim: int = 64,
        features_tranforemer_input_dim: int = 64,
        nodes_tranforemer_hidden_dim: int = 256,
        edges_tranforemer_hidden_dim: int = 128,
        features_tranforemer_hidden_dim: int = 2048
    ):
        super().__init__()
        self.nodes_input_dim = nodes_input_dim
        self.edges_input_dim = edges_input_dim
        self.features_input_dim = features_input_dim

        self.nodes_input_net = nn.Sequential(
            nn.Linear(nodes_input_dim, nodes_hidden_dim), 
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim, nodes_tranforemer_input_dim), 
            nn.ReLU()
        )
        self.edges_input_net = nn.Sequential(
            nn.Linear(edges_input_dim, edges_hidden_dim), 
            nn.ReLU(),
            nn.Linear(edges_hidden_dim, edges_tranforemer_input_dim), 
            nn.ReLU(),
        )
        self.features_input_net = nn.Sequential(
            nn.Linear(features_input_dim, features_hidden_dim), 
            nn.ReLU(),
            nn.Linear(features_hidden_dim, features_tranforemer_input_dim), 
            nn.ReLU(),
        )

        self.transformer = nn.ModuleList(
            [
                TransformerLayer(
                    nodes_input_dim=nodes_tranforemer_input_dim,
                    edges_input_dim=edges_tranforemer_input_dim,
                    features_input_dim=features_tranforemer_input_dim,
                    num_heads=num_heads,
                    nodes_hidden_dim=nodes_tranforemer_hidden_dim,
                    edges_hidden_dim=edges_tranforemer_hidden_dim,
                    features_hidden_dim=features_tranforemer_hidden_dim,
                ) for _ in range(num_layers)
            ]
        )

        self.nodes_output_net = nn.Sequential(
            nn.Linear(nodes_tranforemer_input_dim, nodes_hidden_dim), 
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim, nodes_input_dim), 
            nn.ReLU()
        )
        self.edges_output_net = nn.Sequential(
            nn.Linear(edges_tranforemer_input_dim, edges_hidden_dim), 
            nn.ReLU(),
            nn.Linear(edges_hidden_dim, edges_input_dim), 
            nn.ReLU(),
        )
        self.features_output_net = nn.Sequential(
            nn.Linear(features_tranforemer_input_dim, features_hidden_dim), 
            nn.ReLU(),
            nn.Linear(features_hidden_dim, features_input_dim), 
            nn.ReLU(),
        )

    def forward(
        self, 
        nodes: torch.Tensor, 
        edges: torch.Tensor, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, n = nodes.shape[0], nodes.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(edges).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        nodes_to_out = nodes[..., :self.nodes_input_dim]
        edges_to_out = edges[..., :self.edges_input_dim]
        features_to_out = features[..., :self.features_input_dim]

        new_nodes = self.nodes_input_net(nodes)
        new_edges = self.edges_input_net(edges)
        new_edges = (new_edges + new_edges.transpose(1, 2)) / 2
        new_features = self.features_input_net(features)

        (new_nodes, new_edges), new_features = apply_mask(new_nodes, new_edges, mask), new_features

        for layer in self.transformer:
            new_nodes, new_edges, new_features = layer(new_nodes, new_edges, new_features, mask)

        new_nodes = self.nodes_output_net(new_nodes)
        new_edges = self.edges_output_net(new_edges)
        new_features = self.features_output_net(new_features)

        new_nodes = (new_nodes + nodes_to_out)
        new_edges = (new_edges + edges_to_out) * diag_mask
        new_features = new_features + features_to_out

        new_edges = 1/2 * (new_edges + torch.transpose(new_edges, 1, 2))
        (new_nodes, new_edges), new_features = apply_mask(new_nodes, new_edges, mask), new_features
        return new_nodes, new_edges, new_features

# class DiGress(nn.Module):
#     def __init__(
#         self,
#         model: GraphTransformer,
#         prior: Prior,
#     ) -> None:
#         super().__init__()
#         self.model = model
#         self.prior = prior
#         self.num_node_categories = model.nodes_input_dim
#         self.num_edge_categories = model.edges_input_dim
#         self.num_timesteps = model.num_timesteps
#         self.eps = 1e-6

#     def forward(self, data: MoleculeBatch, t: torch.Tensor) -> MoleculeBatch:
#         (nodes, edges, mask), features = data.to_dense(), data.y
#         features = torch.concat([features, t], dim=-1)
#         nodes, edges, features = self.model(nodes, edges, features, mask)
#         batch = MoleculeBatch(nodes, edges, features, mask)
#         return batch

#     def markov_sample(self, data: MoleculeBatch, t: torch.Tensor) -> MoleculeBatch:
#         """Samples from $p(x_{t-1} | x_{t}, \hat{x_{0}})$, where $\hat{x_{0}} \sim m_{\theta}(\hat{x_{0}} | x_{t})$.""" # type: ignore
#         not_first_step = (t != 1).float().reshape((data.shape[0], *[1] * (data.dim())))

#         nodes_noise = torch.rand(data.x.shape, device=self.device)
#         nodes_noise = torch.clip(nodes_noise, self.eps, 1.0)
#         gumbel_nodes_noise = -torch.log(-torch.log(nodes_noise))
        
        
#         pred_x_end_logits = self(data, t)
#         pred_q_posterior_logits = self.prior.q_posterior_logits(pred_x_end_logits, data, t)
#         # remove gumbel noise at `t==1` because we already have correct logits
#         torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)
#         return MoleculeBatch(nodes, edges, features, mask)
        
#     @torch.no_grad()
#     def sample(self, data: MoleculeBatch) -> MoleculeBatch:
#         for t in reversed(range(1, self.num_timesteps + 2)):
#             t = torch.tensor([t] * data.shape[0], device=self.device)
#             data = self.markov_sample(data, t)
#         return data
    
#     @torch.no_grad()
#     def sample_trajectory(self, data: torch.Tensor) -> List[MoleculeBatch]:
#         trajectory = [data]
#         for t in reversed(range(1, self.num_timesteps + 2)):
#             t = torch.tensor([t] * data.shape[0], device=self.device)
#             features = 
#             data = self.markov_sample(data, features)
#             trajectory.append(data)
#         return trajectory
    
#     @property
#     def device(self):
#         return next(self.parameters()).device