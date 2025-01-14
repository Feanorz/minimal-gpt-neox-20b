import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.multiprocessing import Process, Queue
import time
import copy
import minimal20b.rotary as rotary


# Maps map_fn, onto args, n times
# Watch out for:
#   Exiting process too quickly before tensor has been copied out
#   Shared memory usage too high
def q_map(map_fn, n, args):
    procs = []
    for i in range(n):
        # Make sure main process has enough time to load back output before destroying this function
        mp_queue, sync_queue = Queue(), Queue()

        proc = Process(target=execute_fun, args=(mp_queue, sync_queue, map_fn, args))
        proc.start()

        procs.append((proc, mp_queue, sync_queue))

    results = []
    for proc, mp_queue, sync_queue in procs:
        # Save shared memory when using DOCKER
        result = mp_queue.get().to("meta")
        results.append(copy.deepcopy(result))

        sync_queue.put(1)
        del mp_queue, sync_queue

    return results


def execute_fun(mp_queue, sync_queue, map_fn, args):
    output = map_fn(*args)
    mp_queue.put(output.half())

    # Close function
    sync_queue.get()

    return


class NeoX20BModel(nn.Module):
    def __init__(self, args, use_cache=False, device=torch.device("cpu")):
        super().__init__()
        self.half_precision = args.half_precision
        self.use_cache = use_cache
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size, device=device)

        # Multiprocessing init of transformer layers
        self.layer_list = nn.ModuleList([])

        layers = range(args.num_layers)
        for i in range(0, args.num_layers, args.start_cpu_threads):
            map_size = len(layers[i:i + args.start_cpu_threads])
            print("\n Starting transformer layer", i)
            layer_objects = q_map(TransformerLayer, map_size, (args, use_cache))

            for layer in layer_objects:
                self.layer_list.append(layer)

        self.final_layer_norm = nn.LayerNorm(
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

        self.second_layer_list = None
        self.dynamic_precision = args.dynamic_precision
        self.empty_layer = None

        self.use_gpu = args.use_gpu
        self.full_gpu = args.full_gpu
        self.gpu_layers = args.gpu_layers


    def forward(self, x, attention_mask=None, layer_past=None):

        if self.use_gpu:
            init_device = torch.device("cuda:0")
        else:
            init_device = torch.device("cpu")

        if attention_mask is None:
            attention_mask = generate_mask(x.shape[1], init_device)
        if self.use_cache:
            if layer_past is None:
                kv_length = x.shape[1]
            else:
                kv_length = layer_past[0].shape[1] + 1
            attention_mask = attention_mask[..., :x.shape[1], :kv_length]

        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        hidden_states = self.embed_in(x)
        hidden_states = self.pre_transformer_transpose(hidden_states)

        if self.gpu_layers != 0:
            hidden_states = hidden_states.to(init_device).type(torch.float16)

        for layer_i in range(len(self.layer_list)):
            #print(layer_i)

            # # Complete rest of forward pass on CPU
            if not self.full_gpu and self.gpu_layers and layer_i == self.gpu_layers:
                hidden_states = hidden_states.to(device=torch.device("cpu"), dtype=torch.float32)
                attention_mask = attention_mask.to(torch.device("cpu"))

            layer = self._load_layer(layer_i)

            hidden_states, kv_cache = layer(
                x=hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past[layer_i],
            )
            kv_cache_list.append(kv_cache)

            self._return_layer(layer, layer_i)


        if self.full_gpu:
            hidden_states = hidden_states.to(device=torch.device("cpu"), dtype=torch.float32)

        hidden_states = self.post_transformer_transpose(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.logits_out(hidden_states)

        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    def _load_layer(self, layer_i):
        layer = self.layer_list[layer_i]

        if self.use_gpu:
            if self.full_gpu and layer_i >= self.gpu_layers:
                saved_layer = self.second_layer_list[layer_i]
                for name, param in layer.named_parameters():
                    param.data = saved_layer[name].to(torch.device("cuda:0"))
                for name, buff in layer.named_buffers():
                    buff.data = saved_layer[name].to(torch.device("cuda:0"))

        elif self.dynamic_precision:
            # saved_layer = self.second_layer_list[layer_i]
            # for name, param in layer.named_parameters():
            #     param.data = saved_layer[name].float()
            # for name, buff in layer.named_buffers():
            #     buff.data = saved_layer[name].float()

            layer.load_state_dict(self.second_layer_list[layer_i])
        elif self.half_precision:
            layer.float()
        return layer

    def _return_layer(self, layer, layer_i):
        if self.use_gpu:
            if self.full_gpu and layer_i >= self.gpu_layers:
                #layer.to(device=torch.device("cpu"), non_blocking=True)
                layer.to_empty(device=torch.device("cpu"))
        elif self.dynamic_precision:
            # for name, param in layer.named_parameters():
            #     param.data = self.empty_layer[name]
            layer.to_empty(device=torch.device("cpu"))
        elif self.half_precision:
            layer.half()
        #return layer_i


    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, args, use_cache, device="cpu"):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)
        print("Transformer Layer Completed")

    def forward(self, x, attention_mask, layer_past=None):
        residual = x
        ln_output = self.input_layernorm(x)
        attention_output, kv_cache = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
        )
        post_attn_ln = self.post_attention_layernorm(x)
        mlp_output = self.mlp(hidden_states=post_attn_ln)
        output = residual + mlp_output + attention_output
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
        qkv = self.query_key_value(hidden_states)

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
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

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
        output = self.dense(context_layer)
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
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

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
    val = 0.79788456 * x * (1 + 0.044715 * x * x)
    return x * 0.5 * (1.0 + torch.tanh(val))


class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args.hidden_size
        self.dense_h_to_4h = nn.Linear(args.hidden_size, ff_dim, device=device)
        self.dense_4h_to_h = nn.Linear(ff_dim, args.hidden_size, device=device)

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


def generate_mask(seq_len, device):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device))


def attention_mask_func(attention_scores, ltor_mask):
    """Assign -10000.0 to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, -10000.0)
    return attention_scores
