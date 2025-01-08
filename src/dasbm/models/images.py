from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dasbm.data import Prior


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class TransformerPositionalEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension: int, max_timesteps: int = 1000):
        super(TransformerPositionalEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)
        self.final = nn.Sequential(
            nn.Linear(dimension, dimension * 4), 
            Swish(),
            nn.Linear(dimension * 4, dimension * 4)
        )
        self.register_buffer("pe_matrix", pe_matrix)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # [bs, d_model]
        emb = self.pe_matrix[timestep].to(timestep.device)
        return self.final(emb)


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8, dropout: Optional[float] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = Swish()
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class ResNetBlock(nn.Module):
    """
    In the original DDPM paper Wide ResNet was used
    (https://arxiv.org/pdf/1605.07146.pdf).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        *, 
        time_emb_channels: Optional[int] = None, 
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super(ResNetBlock, self).__init__()
        self.time_embedding_proj = None
        if time_emb_channels is not None:
            self.time_embedding_proj = nn.Sequential(Swish(), nn.Linear(time_emb_channels, out_channels))

        self.block1 = Conv2d(in_channels, out_channels, groups=num_groups)
        self.block2 = Conv2d(out_channels, out_channels, groups=num_groups, dropout=dropout)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_tensor = x
        h = self.block1(x)
        # According to authors implementations, they inject timestep embedding into the network
        # using MLP after the first conv block in all the ResNet blocks
        # (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L49)

        if self.time_embedding_proj is not None and time_embedding is not None:
            time_emb = self.time_embedding_proj(time_embedding)
            time_emb = time_emb[:, :, None, None]
            x = time_emb + h
        else: 
            x = h

        x = self.block2(x)
        return x + self.residual_conv(input_tensor)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, padding: int):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(input_tensor)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 2.0):
        super(UpsampleBlock, self).__init__()

        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # align_corners=True for potential convertibility to ONNX
        x = F.interpolate(input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = self.conv(x)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention blocks are applied at the 16x16 resolution in the original DDPM paper.
    Implementation is based on "Attention Is All You Need" paper, Vaswani et al., 2015
    (https://arxiv.org/pdf/1706.03762.pdf)
    """
    def __init__(self, num_heads: int, in_channels: int, num_groups: int = 32, embedding_dim: int = 256):
        super(SelfAttentionBlock, self).__init__()
        # For each of heads use d_k = d_v = d_model / num_heads
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        self.d_values = embedding_dim // num_heads

        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)

        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def split_features_for_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # We receive Q, K and V at shape [batch, h*w, embedding_dim].
        # This method splits embedding_dim into 'num_heads' features so that
        # each channel becomes of size embedding_dim / num_heads.
        # Output shape becomes [batch, num_heads, h*w, embedding_dim/num_heads],
        # where 'embedding_dim/num_heads' is equal to d_k = d_k = d_v = sizes for
        # K, Q and V respectively, according to paper.
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1)
        heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
        return heads_splitted_tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        batch, features, h, w = x.shape
        # Do view and transpose input tensor since we want to process depth feature maps, not spatial maps
        x = x.view(batch, features, h * w).transpose(1, 2)

        # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
        queries = self.query_projection(x)  # [b, in_channels, embedding_dim]
        keys = self.key_projection(x)       # [b, in_channels, embedding_dim]
        values = self.value_projection(x)   # [b, in_channels, embedding_dim]

        # Split Q, K, V between attention heads to process them simultaneously
        queries = self.split_features_for_heads(queries)
        keys = self.split_features_for_heads(keys)
        values = self.split_features_for_heads(values)

        # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
        # Each SDPA block yields tensor of size d_v = embedding_dim/num_heads.
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, values)

        # Permute computed attention scores such that
        # [batch, num_heads, h*w, embedding_dim] --> [batch, h*w, num_heads, d_v]
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # Concatenate scores per head into one tensor so that
        # [batch, h*w, num_heads, d_v] --> [batch, h*w, num_heads*d_v]
        concatenated_heads_attention_scores = attention_scores.view(batch, h * w, self.d_model)

        # Perform linear projection and view tensor such that
        # [batch, h*w, d_model] --> [batch, d_model, h*w] -> [batch, d_model, h, w]
        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).view(batch, self.d_model, h, w)

        # Residual connection + norm
        x = self.norm(linear_projection + input_tensor)
        return x


class ConvDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        dropout: float,
        num_att_heads: Optional[int] = None,
        downsample: bool = True
    ):
        super(ConvDownBlock, self).__init__()
        self.has_attention = num_att_heads is not None
        self.num_layers = num_layers
        self.in_channels = in_channels

        resnet_blocks = []
        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                dropout=dropout,
                num_groups=num_groups
            )
            resnet_blocks.append(resnet_block)

        if self.has_attention: 
            attention_blocks = []
            for i in range(num_layers):
                in_channels = self.in_channels if i == 0 else out_channels
                attention_block = SelfAttentionBlock(
                    in_channels=out_channels,
                    embedding_dim=out_channels,
                    num_heads=num_att_heads, # type: ignore
                    num_groups=num_groups
                )
                attention_blocks.append(attention_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        if self.has_attention:
            self.attention_blocks = nn.ModuleList(attention_blocks)

        self.downsample = None
        if downsample:  
            self.downsample = DownsampleBlock(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1)
            

    def forward(self, input_tensor: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        for i in range(self.num_layers):
            x = self.resnet_blocks[i](x, time_embedding)
            if self.has_attention:
                x = self.attention_blocks[i](x)

        if self.downsample:
            x = self.downsample(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        dropout: float,
        num_att_heads: Optional[int] = None,
        upsample: bool = True
    ):
        super(ConvUpBlock, self).__init__()
        self.has_attention = num_att_heads is not None
        self.num_layers = num_layers
        self.in_channels = in_channels

        resnet_blocks = []
        
        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                dropout = dropout,
                num_groups=num_groups
            )
            resnet_blocks.append(resnet_block)

        if self.has_attention: 
            attention_blocks = []
            for i in range(num_layers):
                in_channels = self.in_channels if i == 0 else out_channels
                attention_block = SelfAttentionBlock(
                    in_channels=out_channels,
                    embedding_dim=out_channels,
                    num_heads=num_att_heads, # type: ignore
                    num_groups=num_groups
                )
                attention_blocks.append(attention_block)

            

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        if self.has_attention:
            self.attention_blocks = nn.ModuleList(attention_blocks)

        self.upsample = None
        if upsample:  
            self.upsample = UpsampleBlock(in_channels=out_channels, out_channels=out_channels)
            

    def forward(self, input_tensor: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        for i in range(self.num_layers):
            x = self.resnet_blocks[i](x, time_embedding)
            if self.has_attention:
                x = self.attention_blocks[i](x)

        if self.upsample:
            x = self.upsample(x)
        return x


class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    """

    def __init__(
        self, 
        img_size: int = 32,
        num_categories: int = 256,
        in_channels: int = 3, 
        num_channels: int = 64, 
        num_layers: int = 2,
        ch_mults: Tuple[int, ...] = (1, 2, 2, 2),
        attention_resolution: int = 16,
        num_groups: int = 32,
        num_att_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. We replaced weight normalization with group normalization
        # 2. Our 32x32 models use four feature map resolutions (32x32 to 4x4), and our 256x256 models use six (I made 5)
        # 3. Two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution
        # between the convolutional blocks [https://arxiv.org/pdf/1712.09763.pdf]
        # 4. Diffusion time t is specified by adding the Transformer sinusoidal position embedding into
        # each residual block [https://arxiv.org/pdf/1706.03762.pdf]
        n_resolutions = len(ch_mults)
        attention_block_index = img_size // int(attention_resolution)
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_categories = num_categories

        self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = TransformerPositionalEmbedding(dimension=num_channels)

        self.downsample_blocks = []
        out_channels = in_channels = num_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            self.downsample_blocks.append(
                ConvDownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    num_groups=num_groups, 
                    time_emb_channels=num_channels * 4,
                    num_att_heads=num_att_heads if 2 ** i == attention_block_index else None,
                    dropout=dropout
                )
            )
            in_channels = out_channels
        self.downsample_blocks = nn.ModuleList(self.downsample_blocks)
        
        self.bottleneck = ConvDownBlock(
            in_channels=out_channels, 
            out_channels=out_channels, 
            num_layers=num_layers, 
            num_att_heads=num_att_heads, 
            num_groups=num_groups, 
            time_emb_channels=num_channels * 4, 
            dropout=dropout,
            downsample=False
        )
        self.upsample_blocks = []
        

        for i in reversed(range(n_resolutions)):
            out_channels = in_channels // ch_mults[i]
            self.upsample_blocks.append(
                ConvUpBlock(
                    in_channels=in_channels + in_channels,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    num_groups=num_groups, 
                    time_emb_channels=num_channels * 4,
                    num_att_heads=num_att_heads if 2 ** i == attention_block_index else None,
                    dropout=dropout
                )
            )
            in_channels = out_channels
        self.upsample_blocks = nn.ModuleList(self.upsample_blocks)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=num_channels + num_channels, num_groups=num_groups),
            Swish(),
            nn.Conv2d(num_channels + num_channels, self.in_channels * self.num_categories, 3, padding=1)
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)
        initial_x = self.initial_conv(input_tensor.float())

        states_for_skip_connections = [initial_x]

        x = initial_x
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))

        x = self.bottleneck(x, time_encoded)
        for block, skip in zip(self.upsample_blocks, states_for_skip_connections):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        # Get initial shape [3 * 256, 32, 32] with convolutions
        out = self.output_conv(x)
        out = out.view(x.shape[0], self.in_channels, self.img_size, self.img_size, self.num_categories)

        return out
    

class ImageD3PM(nn.Module):
    def __init__(
        self, 
        img_size: int = 32,
        num_categories: int = 256,
        num_timesteps: int = 100, 
        in_channels: int = 3, 
        num_channels: int = 64, 
        num_layers: int = 2,
        ch_mults: Tuple[int, ...] = (1, 2, 2, 2),
        attention_resolution: int = 16,
        num_groups: int = 32,
        num_att_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = UNet(
            img_size,
            num_categories,
            in_channels,
            num_channels,
            num_layers,
            ch_mults,
            attention_resolution,
            num_groups,
            num_att_heads,
            dropout
        )
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_one_hot = F.one_hot(x, self.num_categories) 
        mean = (self.num_categories - 1) / 2
        x = x / mean - 1
        return self.model(x, t) + x_one_hot

    def markov_sample(self, x: torch.Tensor, t: torch.Tensor, prior: Prior):
        r"""Samples from $p(x_{t-1} | x_{t}, \hat{x_{0}})$, where $\hat{x_{0}} \sim m_{\theta}(\hat{x_{0}} | x_{t})$."""
        first_step = (t == 1).long().view((x.shape[0], *[1] * (x.dim() - 1)))
        
        pred_x_end_logits = self(x, t)
        pred_q_posterior_logits = prior.posterior_logits(pred_x_end_logits, x, t, logits=True)
        noise = torch.rand_like(pred_q_posterior_logits)
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        random_samples = torch.argmax(pred_q_posterior_logits + gumbel_noise, dim=-1)
        # probs = pred_q_posterior_logits.softmax(dim=-1).view(-1, self.num_categories)
        # random_samples = probs.multinomial(num_samples=1).view(x.shape)
        
        # No noise when t == 1
        # NOTE: for t=1 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case.

        argmax_samples = pred_q_posterior_logits.argmax(dim=-1)
        samples = first_step * argmax_samples + (1 - first_step) * random_samples
        return samples
        
    @torch.no_grad()
    def sample(self, x: torch.Tensor, prior: Prior) -> torch.Tensor:
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, prior)
        return x
    
    @torch.no_grad()
    def sample_trajectory(self, x: torch.Tensor, prior: Prior) -> torch.Tensor:
        trajectory = [x]
        for t in reversed(range(1, self.num_timesteps + 2)):
            t = torch.tensor([t] * x.shape[0], device=self.device)
            x = self.markov_sample(x, t, prior)
            trajectory.append(x)
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory
    
    @property
    def device(self):
        return next(self.parameters()).device