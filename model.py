import torch
import numpy as np
import coremltools as ct
from torch import nn, Tensor
from torch.nn import functional as F

seq_len = 384
num_heads = 12
num_layers = 6
batch_size = 8
hidden_size = 768
vocab_size = 50265
intermediate_size = hidden_size * 4
head_dim = hidden_size // num_heads

mask = torch.triu(torch.ones(seq_len, seq_len, device="mps"), 1)[None, :, None, :] * -1e4

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(batch_size, hidden_size, 1, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(batch_size, hidden_size, 1, 1, dtype=torch.float16))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        zero_mean = x - mean
        denom = (zero_mean.pow(2).mean(1, keepdim=True) + 1e-05).rsqrt()
        return (zero_mean * denom * self.weight + self.bias).to(torch.float16)

class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size, dtype=torch.float16)
        self.token_type_embeddings = nn.Embedding(seq_len, hidden_size, dtype=torch.float16)
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings.weight
        position_embeddings = self.position_embeddings.weight
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        # BSC -> BC1S
        embeddings = torch.transpose(embeddings, 1, 2).unsqueeze(2)
        return self.layer_norm(embeddings)
    
class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True, dtype=torch.float16)
        self.k_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True, dtype=torch.float16)
        self.v_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True, dtype=torch.float16)
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=False, dtype=torch.float16)

    def forward(self, hidden_states: Tensor) -> Tensor:
        query_state = torch.reshape(self.q_proj(hidden_states), (batch_size, head_dim, num_heads, seq_len))
        key_state = torch.reshape(self.k_proj(hidden_states), (batch_size, head_dim, num_heads, seq_len))
        value_state = torch.reshape(self.v_proj(hidden_states), (batch_size, head_dim, num_heads, seq_len))
        
        query_states = torch.split(query_state, 1, dim=2)
        key_states = torch.split(torch.transpose(key_state, 1, 3), 1, dim=2)
        value_states = torch.split(value_state, 1, dim=2)

        weights = [torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * float(head_dim) ** -0.5 for qi, ki in zip(query_states, key_states)]
        weights = [torch.softmax(w + mask, dim=1, dtype=torch.float32) for w in weights]

        attn = [torch.einsum("bkhq,bchk->bchq", [wi, vi]) for wi, vi in zip(weights, value_states)]
        attn = torch.cat(attn, dim=1).to(torch.float16)

        return self.o_proj(attn)
    
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, 1, dtype=torch.float16)
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, 1, dtype=torch.float16)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))
    
class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = MLP()
        self.attention = Attention()
        self.post_attention_layernorm = LayerNorm(hidden_size)
        self.post_mlp_layernorm = LayerNorm(hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states += self.attention(hidden_states)
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
        return self.post_mlp_layernorm(hidden_states)
    
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
        self.dense = nn.Conv2d(hidden_size, hidden_size, 1, dtype=torch.float16)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return torch.reshape(F.tanh(self.dense(hidden_states[:, :, :, :1])), (batch_size, hidden_size))
    
model = Model().to(torch.float16).to("mps").eval()
print(f"number of parameters {sum(p.numel() for p in model.parameters()):,d}")

with torch.no_grad():
    traced_model = torch.jit.trace(model, (torch.randint(0, vocab_size, (batch_size, seq_len), device="mps")))

coreml_model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(batch_size, seq_len), dtype=np.int32)
    ],
    outputs=[
        ct.TensorType(name="outputs")
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
    skip_model_load=True,
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

coreml_model.save("AneEmbeddings.mlpackage")