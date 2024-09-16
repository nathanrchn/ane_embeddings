import torch
from transformers import BertModel

def load_model(url: str, num_layers: int) -> dict:
    model = BertModel.from_pretrained(url, add_pooling_layer=False).to(torch.float16).to("mps").eval()
    state_dict = model.state_dict()
    
    # state_dict = load_file("model.safetensors")
    # for k, v in state_dict.items():
    #     state_dict[k] = v.to(torch.float16).to("mps")

    new_state_dict = {}

    new_state_dict["embeddings.word_embeddings.weight"] = state_dict["embeddings.word_embeddings.weight"]
    new_state_dict["embeddings.position_embeddings.weight"] = state_dict["embeddings.position_embeddings.weight"]
    new_state_dict["embeddings.token_type_embeddings.weight"] = state_dict["embeddings.token_type_embeddings.weight"][0].repeat(512, 1)
    new_state_dict["embeddings.layer_norm.weight"] = state_dict["embeddings.LayerNorm.weight"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    new_state_dict["embeddings.layer_norm.bias"] = state_dict["embeddings.LayerNorm.bias"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

    for i in range(num_layers):
        new_state_dict[f"layers.{i}.mlp.fc1.weight"] = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.mlp.fc1.bias"] = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"]
        new_state_dict[f"layers.{i}.mlp.fc2.weight"] = state_dict[f"encoder.layer.{i}.output.dense.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.mlp.fc2.bias"] = state_dict[f"encoder.layer.{i}.output.dense.bias"]
        new_state_dict[f"layers.{i}.attention.q_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.query.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.attention.q_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.query.bias"]
        new_state_dict[f"layers.{i}.attention.k_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.key.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.attention.k_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.key.bias"]
        new_state_dict[f"layers.{i}.attention.v_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.value.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.attention.v_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.value.bias"]
        new_state_dict[f"layers.{i}.attention.o_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].unsqueeze(-1).unsqueeze(-1)
        new_state_dict[f"layers.{i}.attention.o_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"]
        new_state_dict[f"layers.{i}.post_attention_layernorm.weight"] = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        new_state_dict[f"layers.{i}.post_attention_layernorm.bias"] = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        new_state_dict[f"layers.{i}.post_mlp_layernorm.weight"] = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        new_state_dict[f"layers.{i}.post_mlp_layernorm.bias"] = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

    # print the shape of the weights
    # for k, v in new_state_dict.items():
    #     print(k, v.shape, v.dtype)

    # new_state_dict["dense.weight"] = state_dict["pooler.dense.weight"].unsqueeze(-1).unsqueeze(-1)
    # new_state_dict["dense.bias"] = state_dict["pooler.dense.bias"]

    return new_state_dict

def load_modelo(url: str, num_layers: int) -> dict:
    model = BertModel.from_pretrained(url).to(torch.float16).to("mps").eval()
    state_dict = model.state_dict()
    
    # state_dict = load_file("model.safetensors")
    # for k, v in state_dict.items():
    #     state_dict[k] = v.to(torch.float16).to("mps")

    new_state_dict = {}

    new_state_dict["embeddings.word_embeddings.weight"] = state_dict["embeddings.word_embeddings.weight"]
    new_state_dict["embeddings.position_embeddings.weight"] = state_dict["embeddings.position_embeddings.weight"]
    new_state_dict["embeddings.token_type_embeddings.weight"] = state_dict["embeddings.token_type_embeddings.weight"][0].repeat(512, 1)
    new_state_dict["embeddings.layer_norm.weight"] = state_dict["embeddings.LayerNorm.weight"]
    new_state_dict["embeddings.layer_norm.bias"] = state_dict["embeddings.LayerNorm.bias"]

    for i in range(num_layers):
        new_state_dict[f"layers.{i}.mlp.fc1.weight"] = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"]
        new_state_dict[f"layers.{i}.mlp.fc1.bias"] = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"]
        new_state_dict[f"layers.{i}.mlp.fc2.weight"] = state_dict[f"encoder.layer.{i}.output.dense.weight"]
        new_state_dict[f"layers.{i}.mlp.fc2.bias"] = state_dict[f"encoder.layer.{i}.output.dense.bias"]
        new_state_dict[f"layers.{i}.attention.q_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.query.weight"]
        new_state_dict[f"layers.{i}.attention.q_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.query.bias"]
        new_state_dict[f"layers.{i}.attention.k_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.key.weight"]
        new_state_dict[f"layers.{i}.attention.k_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.key.bias"]
        new_state_dict[f"layers.{i}.attention.v_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.self.value.weight"]
        new_state_dict[f"layers.{i}.attention.v_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.self.value.bias"]
        new_state_dict[f"layers.{i}.attention.o_proj.weight"] = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"]
        new_state_dict[f"layers.{i}.attention.o_proj.bias"] = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"]
        new_state_dict[f"layers.{i}.post_attention_layernorm.weight"] = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"]
        new_state_dict[f"layers.{i}.post_attention_layernorm.bias"] = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"]
        new_state_dict[f"layers.{i}.post_mlp_layernorm.weight"] = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"]
        new_state_dict[f"layers.{i}.post_mlp_layernorm.bias"] = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"]

    return new_state_dict
