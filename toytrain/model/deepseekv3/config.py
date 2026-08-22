from dataclasses import dataclass, field
from typing import Literal, Tuple

import torch.nn as nn
from loguru import logger


@dataclass
class MLAConfig:
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128


@dataclass
class YaRNConfig:
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


@dataclass
class MoERouterConfig:
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True


@dataclass
class MoETokenChoiceConfig:
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3


@dataclass
class MoEConfig:
    num_experts: int = 8
    num_shared_experts: int = 1
    router_config: MoERouterConfig = field(default_factory=MoERouterConfig)
    token_choice_config: MoETokenChoiceConfig = field(
        default_factory=MoETokenChoiceConfig
    )


@dataclass
class DeepSeekV3Config:
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm
    moe_config: MoEConfig = field(default_factory=MoEConfig)
    mla_config: MLAConfig = field(default_factory=MLAConfig)
    yarn_config: YaRNConfig = field(default_factory=YaRNConfig)

    def get_num_params_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> Tuple[int, int]:
        (
            num_params_embedding,
            num_params_moe_router,
            num_params_shared_experts,
            num_params_experts,
            num_params_dense,
        ) = [0] * 5

        for name, param in model.named_parameters():
            if "embedding" in name:
                num_params_embedding + param.numel()
                num_params_dense += param.numel()
            elif "moe.shared_experts" in name:
                num_params_shared_experts += param.numel()
            elif "moe.router" in name:
                num_params_moe_router += param.numel()
            else:
                num_params_dense += param.numel()

        num_params_sparse = (
            num_params_shared_experts + num_params_experts + num_params_moe_router
        )
        num_total_params = num_params_dense + num_params_sparse
        top_k = self.moe_config.token_choice_config.top_k
        num_params_sparse_active = (
            num_params_moe_router
            + num_params_shared_experts
            + num_params_experts * top_k // self.moe_config.num_experts
        )
        num_params_active = num_params_dense + num_params_sparse_active

        logger.info(
            f"Total param count dense: {num_params_dense}, "
            f"Total param count sparse: {num_params_sparse}, "
            f"Total param count active: {num_params_active}, "
            f"Total param count: {num_total_params}."
        )

        head_dims = (
            self.mla_config.qk_nope_head_dim
            + self.mla_config.qk_rope_head_dim
            + self.mla_config.v_head_dim
        )
        num_flops_per_token = (
            6 * (num_params_dense - num_params_embedding + num_params_sparse_active)
            + 6 * self.n_layers * self.n_heads * head_dims * seq_len
        )

        return num_total_params, num_flops_per_token
