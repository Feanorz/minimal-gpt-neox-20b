import torch.nn as nn
import torch
import math
from torch import Tensor, Size
from typing import Union, List, Tuple
import torch.nn.init as init
import numbers
from torch.multiprocessing import Pool


import minimal20b.rotary as rotary
TORCH_DTYPE = torch.float32

class NeoX20BModel(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        
        # Multiprocessing init of transformer layers
        layer_inputs = [(args, False) for i in range(args.num_layers)]
        with Pool(32) as p:
            transformer_layers = p.starmap(TransformerLayer, layer_inputs)
            
        self.layer_list = nn.ModuleList([])
        for transformer_layer in transformer_layers:
            #l = TransformerLayer(args, use_cache, device=device)
            self.layer_list.append(transformer_layer)
        

        self.final_layer_norm = CPULayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.logits_out = nn.Linear(
            args.hidden_size,
            args.vocab_size,
            bias=False,
            device=device,
        )

    def forward(self, x, attention_mask=None, layer_past=None):
        print("Started model forward pass")
        print()
        if attention_mask is None:
            attention_mask = generate_mask(x.shape[1]).to(x.device)
            
        print("Mask Generated")
        
        if self.use_cache:
            if layer_past is None:
                kv_length = x.shape[1]
            else:
                kv_length = layer_past[0].shape[1] + 1
            attention_mask = attention_mask[..., :x.shape[1], :kv_length]
            
        print("Cache Finished")

        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        hidden_states = self.embed_in(x)
        hidden_states = self.pre_transformer_transpose(hidden_states)
        
        print("Hidden state generated")
        print()

        for layer_i, layer in enumerate(self.layer_list):
            print("Started layer:", layer_i)
            hidden_states, kv_cache = layer(
                x=hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past[layer_i],
            )
            kv_cache_list.append(kv_cache)
        hidden_states = self.post_transformer_transpose(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.logits_out(hidden_states)
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, args, use_cache, device=torch.device("cpu")):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = CPULayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = CPULayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)
        print("Transformer Layer Done")

    def forward(self, x, attention_mask, layer_past=None):
        residual = x
        ln_output = self.input_layernorm(x)
        print("Started Attention")
        attention_output, kv_cache = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
        )
        print("Attention Completed")
        post_attn_ln = self.post_attention_layernorm(x)
        print("Started MLP")
        mlp_output = self.mlp(hidden_states=post_attn_ln)
        print("MLP completed")
        output = residual + mlp_output + attention_output
        print("Transformer layer finished")
        print()
        return output, kv_cache


class SelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size_per_attention_head = args.hidden_size // args.num_attention_heads
        self.rotary_ndims = int(self.hidden_size_per_attention_head * args.rotary_pct)

        self.rotary_emb = rotary.RotaryEmbedding(
            self.rotary_ndims,
            base=args.rotary_emb_base,
            device=device,
        )
        self.query_key_value = nn.Linear(
            args.hidden_size,
            3 * args.hidden_size,
            device=device,
        )
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dense = nn.Linear(
            args.hidden_size,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0


        # Compute QKV
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        
        print(" Start Attention Module")
        qkv = self.query_key_value(hidden_states)
        print(" QKV completed")

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.hidden_size_per_attention_head]
        key_layer = qkv[..., self.hidden_size_per_attention_head: 2 * self.hidden_size_per_attention_head]
        value_layer = qkv[..., 2 * self.hidden_size_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
            
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        
        query_layer, key_layer = rotary.apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        print(" Starting attention mechanism")
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )
        print(" Attention mechanism complete")

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        print(" Started final dense layer")
        output = self.dense(context_layer)
        print(" Dense layer finished")

        return output, kv_cache

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores, attention_mask) \
            if attention_mask is not None else attention_scores
            
        # softmax doesn't work with fp16
        softmax = torch.nn.Softmax(dim=-1)
        attention_probs = softmax(masked_scores.type(torch.float32)).type(TORCH_DTYPE)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer



@torch.jit.script
def gelu(x):
    val_16 = 0.79788456 * x * (1 + 0.044715 * x * x)
    val_32 = val_16.type(torch.float32)
    return x * 0.5 * (1.0 + torch.tanh(val_32).type(torch.float16))



class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args.hidden_size
        self.dense_h_to_4h = nn.Linear(args.hidden_size, ff_dim, device=device)
        self.dense_4h_to_h = nn.Linear(ff_dim, args.hidden_size, device=device)

    def forward(self, hidden_states):
        print(hidden_states.shape)
        weight = self.dense_4h_to_h.weight.tolist()
        with open("./weights", "w") as f:
            f.write(str(weight))
        print("Written")
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        print("   1/3")
        intermediate_parallel = gelu(intermediate_parallel)
        print("   2/3")
        output = self.dense_4h_to_h(intermediate_parallel)
        print("   3/3")
        return output




def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))


def attention_mask_func(attention_scores, ltor_mask):
    """Assign -10000.0 to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, -10000.0)
    return attention_scores







# Custom Layer norm to support fp16
_shape_t = Union[int, List[int], Size]
class CPULayerNorm(torch.nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CPULayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    # Custom forward function
    def forward(self, input: Tensor) -> Tensor:
        my_output = self._custom_layer_norm(input, self.normalized_shape)
        return my_output


    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

    # No square root in fp16
    def _custom_layer_norm(self, in_ten: Tensor, dim: Tuple[int], eps: float = 0.00001) -> torch.Tensor:
        dim=(-1)
        in_ten = in_ten.float()
        mean = torch.mean(in_ten, dim=dim, keepdim=True)
        var = torch.square(in_ten - mean).mean(dim=dim, keepdim=True)
        out_ten = (in_ten - mean) / torch.sqrt(var + eps) * self.weight + self.bias
        return out_ten.type(torch.float16)


