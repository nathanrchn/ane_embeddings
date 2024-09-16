import torch
import numpy as np
import coremltools as ct
from shutil import copytree
from torch import nn, Tensor
from torch.nn import functional as F

from load import load_modelo

url = "Snowflake/snowflake-arctic-embed-m-v1.5"
seq_len = 512
num_heads = 12
num_layers = 12
batch_size = 1
batch_slice = 1
hidden_size = 768
vocab_size = 30522
layer_norm_eps = 1e-5
intermediate_size = hidden_size * 4
head_dim = hidden_size // num_heads

class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(seq_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings.weight
        position_embeddings = self.position_embeddings.weight
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        return self.layer_norm(embeddings)
    
class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        query_state = torch.reshape(self.q_proj(hidden_state), (batch_size, seq_len, num_heads, head_dim)).transpose(1, 2)
        key_state = torch.reshape(self.k_proj(hidden_state), (batch_size, seq_len, num_heads, head_dim)).transpose(1, 2)
        value_state = torch.reshape(self.v_proj(hidden_state), (batch_size, seq_len, num_heads, head_dim)).transpose(1, 2)

        attn = F.scaled_dot_product_attention(query_state, key_state, value_state, attn_mask=mask)
        attn = torch.reshape(attn.transpose(1, 2), (batch_size, seq_len, hidden_size))

        output = self.o_proj(attn)
        return output
    
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))
    
class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = MLP()
        self.attention = Attention()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.post_mlp_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        hidden_states += self.attention(hidden_states, mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states += self.mlp(hidden_states)
        return self.post_mlp_layernorm(hidden_states)
    
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        hidden_states = self.embeddings(input_ids)
        mask = (mask[:, None, None, :] - 1.0) * 1e4

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return F.normalize(hidden_states[:, 0], p=2, dim=1)
    
model = Model().to("mps").eval()

model.load_state_dict(load_modelo(url, num_layers))

with torch.no_grad():
    traced_model = torch.jit.trace(
        model,
        (
            torch.randint(0, vocab_size, (batch_size, seq_len), device="mps"),
            torch.zeros((batch_size, seq_len), device="mps", )
        )
    )

coreml_model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(batch_size, seq_len), dtype=np.int32),
        ct.TensorType(name="mask", shape=(batch_size, seq_len), dtype=np.float16)
    ],
    outputs=[
        ct.TensorType(name="outputs")
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

coreml_model.save(f"ane-sdpa-{url.split('/')[-1]}.mlpackage")

# compiled_model_path = coreml_model.get_compiled_model_path()
# print(compiled_model_path)
# copytree(compiled_model_path, f"localy/ane-{url.split('/')[-1]}/model.mlmodelc", dirs_exist_ok=True)
