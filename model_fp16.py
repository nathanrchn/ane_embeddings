import torch
import numpy as np
import coremltools as ct
from shutil import copytree
from torch import nn, Tensor
from torch.nn import functional as F

from load import load_model

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

# mask = torch.triu(torch.ones(seq_len, seq_len, device="mps"), 1)[None, :, None, :] * -1e4

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, hidden_size, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, hidden_size, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 3)
        o = F.layer_norm(x, (hidden_size,), self.weight.view(-1), self.bias.view(-1), layer_norm_eps)
        return o.transpose(1, 3)

class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(seq_len, hidden_size)
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
        self.q_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.k_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.o_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)
        self.v_proj = nn.Conv2d(hidden_size, hidden_size, 1, bias=True)

    def forward(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        masks = torch.split(mask, batch_slice, dim=0)
        hidden_states = torch.split(hidden_state, batch_slice, dim=0)
        query_state = [self.q_proj(hs) for hs in hidden_states]
        key_state = [self.k_proj(hs) for hs in hidden_states]
        value_state = [self.v_proj(hs) for hs in hidden_states]

        query_states = [torch.split(qs, head_dim, dim=1) for qs in query_state]
        key_states = [torch.split(torch.transpose(ks, 1, 3), head_dim, dim=3) for ks in key_state]
        value_states = [torch.split(vs, head_dim, dim=1) for vs in value_state]

        weights = [[torch.einsum("bchq,bkhc->bkhq", (qi, ki)) * head_dim ** -0.5 for qi, ki in zip(qs, ks)] for qs, ks in zip(query_states, key_states)]
        weights = [[F.softmax(w + m, dim=1) for w in wi] for wi, m in zip(weights, masks)]

        attn = [[torch.einsum("bkhq,bchk->bchq", [wi, vi]) for wi, vi in zip(w, vs)] for w, vs in zip(weights, value_states)]
        attn = [torch.cat(a, dim=1) for a in attn]

        outputs = [self.o_proj(a) for a in attn]
        output = torch.cat(outputs, dim=0)
        return output
    
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, intermediate_size, 1)
        self.fc2 = nn.Conv2d(intermediate_size, hidden_size, 1)
        
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
        mask = (mask[:, :, None, None] - 1.0) * 1e4

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return F.normalize(hidden_states[:, :, 0, 0], p=2, dim=1)
    
model = Model().to("mps").eval()

model.load_state_dict(load_model(url, num_layers))

# # DEBUG
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained(url)
# sentences = ["When one of the vectors is all zeros"]
# encoded_input = tokenizer(sentences, padding="max_length", truncation=True, return_tensors='pt', max_length=512)
# with torch.no_grad():
#     outputs = model(encoded_input["input_ids"].to("mps"), encoded_input["attention_mask"].to("mps").to(torch.float16))
# torch.save(outputs, "o1.pth")
# o1 = torch.load("o1.pth").to("cpu")
# o2 = torch.load("o2.pth").to("cpu")
# cs = torch.nn.CosineSimilarity(dim=1)
# print(cs(o1, o2))
# # END DEBUG

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

coreml_model.save(f"ane-{url.split('/')[-1]}.mlpackage")

compiled_model_path = coreml_model.get_compiled_model_path()
print(compiled_model_path)
copytree(compiled_model_path, f"localy/ane-{url.split('/')[-1]}/model.mlmodelc", dirs_exist_ok=True)
