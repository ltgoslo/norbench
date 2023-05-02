from transformers.configuration_utils import PretrainedConfig


class NorT5Config(PretrainedConfig):
    """Configuration class to store the configuration of a `NorT5`.
    """
    def __init__(
        self,
        vocab_size=50000,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_size=768,
        intermediate_size=2048,
        max_position_embeddings=512,
        position_bucket_size=32,
        num_attention_heads=12,
        num_hidden_layers=12,
        layer_norm_eps=1.0e-7,
        output_all_encoded_layers=True,
        pad_token_id=3,
        cls_token_id=1,
        sep_token_id=2,
        bos_token_id=5,
        eos_token_id=6,
        use_cache=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.output_all_encoded_layers = output_all_encoded_layers
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_cache = use_cache
