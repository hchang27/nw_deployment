from params_proto import PrefixProto


class Go1RGBConfig(PrefixProto):
    n_proprio = 53
    n_scan = 132
    num_actions = 12
    scan_encoder_dims = [128, 64, 32]
    actor_hidden_dims = [512, 256, 128]
    priv_encoder_dims = [64, 20]
    estimator_hidden_dims = [128, 64]
    n_priv_latent = 29
    n_priv = 9
    history_len = 10
    depth_buffer_len = 3
    activation_fn = "elu"
    tanh_encoder_output = False
