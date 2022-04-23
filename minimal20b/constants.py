class Args20b:
    vocab_size = 50432
    hidden_size = 6144
    num_attention_heads = 64
    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 44
    # Inference Arguments
    half_precision = False
    dynamic_precision = False
    start_cpu_threads = 32
    gpu_layers = 20



class ArgsDummy:
    vocab_size = 50432
    hidden_size = 64
    num_attention_heads = 4
    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 2

