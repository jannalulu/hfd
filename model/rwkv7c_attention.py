import math
from typing import Optional, Any
import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from .backstepping_longhead import attn_backstepping_longhead

from logger import print0 as print

try:
    from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
except ImportError:
    print("Required module is not installed. Please install it using the following commands:")
    print("pip install flash-linear-attention")
    print("Additionally, ensure you have at least version 2.2.0 of Triton installed:")
    print("pip install triton>=2.2.0")
    raise


def repeat_kv_BTHD(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads along the head dimension (GQA).
    Input:  (B, T, H_kv, D)
    Output: (B, T, H_kv * n_rep, D)
    """
    B, T, H_kv, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand head dim
    hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
    hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
    return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()

class RWKV7cAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # FIXME - this may be different for non-Qwen3 models, maybe we should give our attention sublayer its own config initially populated from the specific teacher to move compatibility modularly outside of this code?
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_dim = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
             
              
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dim = self.num_attention_heads * self.head_dim
 
        print(f'layer = {layer_idx} head_dim {self.head_dim} num_attention_heads {self.num_attention_heads}')

        H = self.num_attention_heads
        N = self.head_dim
        d_att = self.attention_dim
        hidden_dim = self.hidden_dim
        n_layer = self.num_hidden_layers

        # FIXME - find a way to defer initialization via reset_parameters, potentially deferred until before loading but after instantiation
        with torch.no_grad():
            ratio_0_to_1 = layer_idx / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_att)
            for i in range(d_att):
                ddd[0, 0, i] = i / d_att
 
            def ortho_init(x, scale=1.0):
                """
                高速Orthogonal初期化:
                  1. GPU (cuda:0) 上でQR分解を使って直交初期化
                  2. 結果をCPU上のテンソルとして返す
                """
                gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                shape = x.shape
                dim = len(shape)
            
                # QR分解はfloat32で行う（安定性重視）
                def gpu_qr(rows, cols, batch=None):
                    if batch is None:
                        a = torch.randn(max(rows, cols), min(rows, cols), device=gpu_device, dtype=torch.float32)
                        q, _ = torch.linalg.qr(a, mode="reduced")
                        q = q[:rows, :cols] if rows >= cols else q.T[:rows, :cols]
                    else:
                        a = torch.randn(batch, max(rows, cols), min(rows, cols), device=gpu_device, dtype=torch.float32)
                        q, _ = torch.linalg.qr(a, mode="reduced")
                        if rows < cols:
                            q = q.transpose(1, 2)
                        q = q[:, :rows, :cols]
                    return q
            
                if dim == 2:
                    rows, cols = shape
                    gain = math.sqrt(rows / cols) if rows > cols else 1
                    q = gpu_qr(rows, cols)
                    q *= gain * scale
                    # CPUに戻す
                    q_cpu = q.to("cpu", dtype=x.dtype)
                    x.copy_(q_cpu)
            
                elif dim == 3:
                    b, m, n = shape
                    gain = math.sqrt(m / n) if m > n else 1
                    q = gpu_qr(m, n, batch=b)
                    q *= gain * scale
                    q_cpu = q.to("cpu", dtype=x.dtype)
                    x.copy_(q_cpu)
            
                else:
                    # 例外ではなく警告にしておく（落とさない）
                    import warnings
                    warnings.warn(f"[ortho_init_gpu_then_cpu] Unsupported tensor shape {shape}, only 2D/3D supported")
                    return x
            
                # 最終的にCPU上のxを返す
                return x.cpu()

            www = torch.zeros(d_att)
            zigzag = torch.zeros(d_att)
            linear = torch.zeros(d_att)

            for n in range(d_att):
                linear[n] = n / (d_att-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (d_att - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)


            D_DECAY_LORA = max(256, int(round(  (1.7*(d_att**0.6))  /32)*32))
            D_AAA_LORA   = max(128, int(round(  (1.8*(d_att**0.5))  /32)*32)) 
            D_GATE_LORA  = max(256, int(round(  (1.7*(d_att**0.6))  /32)*32))
            D_MV_LORA = max(32, int(round(  (1.3*(d_att**0.5))  /32)*32))
            
            D_MK_LORA = max(32, int(round(  (1.3*(d_att**0.5))  /32)*32)) 
             
            # FIXME - should really be initiaizing w1 a2 v1 as zeros
            
            print(f'D_DECAY_LORA={D_DECAY_LORA}')
            self.w1 = nn.Parameter(ortho_init(torch.zeros(hidden_dim, D_DECAY_LORA),0.1))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, d_att), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,d_att) + 0.5 + zigzag*2.5)
 
            print(f'D_AAA_LORA={D_AAA_LORA}')
            self.a1 = nn.Parameter(ortho_init(torch.zeros(hidden_dim, D_AAA_LORA),0.1))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, d_att), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,d_att)-0.19 + zigzag*0.3 + linear*0.4)

            print(f'D_MV_LORA={D_MV_LORA}')
            self.v1 = nn.Parameter(ortho_init(torch.zeros(hidden_dim, D_MV_LORA),0.1))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, d_att), 0.1))
            self.D_MV_LoRA_Scaling = 0.2
            
            self.g1 = nn.Parameter(torch.zeros(hidden_dim, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, d_att), 0.1))
            print(f'D_GATE_LORA={D_GATE_LORA}')

    @torch.compile
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,

        use_cache: bool,
        hidden_states: torch.Tensor,
        position_embeddings,
        past_key_values,
        cache_position,
        **kwargs
    ):
        x = hidden_states
        B, T, C = x.shape
        B, H, T, N = query.shape
        B, KVH, T, N = key.shape

        # NOTE - would be more efficient if we were the ones applying RoPE here, since the parent model sends us B,H,T,N contiguous not B,T,H,N

        r = query.transpose(1,2).view(B,T,H,N)
        k = repeat_kv_BTHD(key.transpose(1,2).view(B,T,KVH,N), H // KVH).view(B,T,H,N)
        v = repeat_kv_BTHD(value.transpose(1,2).view(B,T,KVH,N), H // KVH).view(B,T,H,N) + ((x @ self.v1 @ self.v2) * self.D_MV_LoRA_Scaling).view(B,T,H,N)

        log_neglog_forget = (-F.softplus(-(self.w0 + F.tanh(x @ self.w1) @ self.w2)) - 0.5).view(B,T,H,N)
        log_forget = -log_neglog_forget.exp()
        forget = log_forget.exp()

        iclr = torch.sigmoid(self.a0 + (x @ self.a1) @ self.a2).view(B,T,H,N)
        g = (F.sigmoid(x @ self.g1) @ self.g2).view(B,T,H,N)

        kk = k.float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).to(k.dtype)
        k = k * (1.0 - forget + iclr)

        # support for left-padding during inference
        if not self.training and attention_mask is not None:
            if len(attention_mask.shape) == 2:
                v = v * attention_mask[:, -T:, None, None]
            elif len(attention_mask.shape) == 4:
                v = v * attention_mask[:, -1, -1, -T:, None, None]

        if use_cache and past_key_values is not None:
            vk_state, shift_state = past_key_values.update(None, None, self.layer_idx)
        #shift_state = shift_state or torch.zeros_like(x[:, -1:])

        z = -kk
        b = kk*iclr
        if self.training:
            x = attn_backstepping_longhead(r, log_neglog_forget, k, v, z, b)[0]
        else:
            x, vk_state = fused_recurrent_rwkv7(r, log_forget, k, v, z, b, scale=1.0, initial_state=vk_state, output_final_state=True, head_first=False)
            #shift_state = x[:, -1:]

        if use_cache and past_key_values is not None:
            past_key_values.update(vk_state, shift_state, self.layer_idx)

        x = x * (N ** -0.5) 
        x = x * g

        return x
