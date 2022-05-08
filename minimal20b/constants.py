class Args20b:
    vocab_size = 50432
    hidden_size = 6144
    num_attention_heads = 64
    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 44

    # Inference Arguments
    start_cpu_threads = 32

    # Standard fp32 CPU
    full_precision_cpu = False
    # Store layers on CPU using half precision, cast to float then back when layer is needed.
    half_precision = False
    # Store layers in separate list, copy layer to model when needed then delete layer in model.
    dynamic_precision = False
    # Load weights using state_dict()
    use_state_dict = False

    use_gpu = True
    # Number of layers to store on GPU
    gpu_layers = 20
    # Store CPU weights using fp16, move to GPU when needed
    full_gpu = True



class ArgsDummy:
    vocab_size = 50432
    hidden_size = 64
    num_attention_heads = 4
    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 2

