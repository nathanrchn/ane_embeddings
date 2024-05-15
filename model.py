import torch
import numpy as np
import coremltools as ct
from torch import nn, Tensor
from torch.nn import functional as F

from load import load_model

url = "silvainrichou/snowflake-arctic-embed-m"
seq_len = 512
num_heads = 12
num_layers = 12
batch_size = 1
batch_slice = 1
hidden_size = 768
vocab_size = 30522
layer_norm_eps = 1e-12
intermediate_size = hidden_size * 4
head_dim = hidden_size // num_heads

# mask = torch.triu(torch.ones(seq_len, seq_len, device="mps"), 1)[None, :, None, :] * -1e4

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, hidden_size, 1, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(1, hidden_size, 1, 1, dtype=torch.float16))

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp_(-250, 250) # for numerical stability
        mean = x.mean(1, keepdim=True)
        zero_mean = x - mean
        denom = (zero_mean.pow(2).mean(1, keepdim=True) + layer_norm_eps).rsqrt()
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
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True, dtype=torch.float16)

    def forward(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        masks = torch.split(mask, batch_slice, dim=0)
        hidden_states = torch.split(hidden_state, batch_slice, dim=0)
        query_state = [torch.reshape(self.q_proj(hs), (batch_slice, head_dim, num_heads, seq_len)) for hs in hidden_states]
        key_state = [torch.reshape(self.k_proj(hs), (batch_slice, head_dim, num_heads, seq_len)) for hs in hidden_states]
        value_state = [torch.reshape(self.v_proj(hs), (batch_slice, head_dim, num_heads, seq_len)) for hs in hidden_states]
        
        query_states = [torch.split(qs, 1, dim=2) for qs in query_state]
        key_states = [torch.split(torch.transpose(ks, 1, 3), 1, dim=2) for ks in key_state]
        value_states = [torch.split(vs, 1, dim=2) for vs in value_state]

        weights = [[torch.einsum("bchq,bkhc->bkhq", (qi, ki)) * float(head_dim) ** -0.5 for qi, ki in zip(qs, ks)] for qs, ks in zip(query_states, key_states)]
        weights = [[torch.softmax(w + m, dim=1) for w in wi] for wi, m in zip(weights, masks)]

        attn = [[torch.einsum("bkhq,bchk->bchq", [wi, vi]) for wi, vi in zip(w, vs)] for w, vs in zip(weights, value_states)]
        attn = [torch.cat(a, dim=1).to(torch.float16) for a in attn]

        outputs = [self.o_proj(a) for a in attn]
        return torch.cat(outputs, dim=0)
    
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
        mask = ((mask - 1) * 1e4).unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return F.normalize(hidden_states[:, :, 0, 0].to(torch.float32), p=2, dim=1)
    
model = Model().to(torch.float16).to("mps").eval()

model.load_state_dict(load_model(url, num_layers))

with torch.no_grad():
    traced_model = torch.jit.trace(
        model,
        (
            torch.randint(0, vocab_size, (batch_size, seq_len), device="mps"),
            torch.zeros((batch_size, seq_len), device="mps", dtype=torch.float16)
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
    minimum_deployment_target=ct.target.iOS17,
    skip_model_load=True,
    compute_units=ct.ComputeUnit.CPU_AND_NE
)

coreml_model.save("AneEmbeddings.mlpackage")