import torch
import torch.nn as nn
import torchvision.models as torch_models

from einops import repeat

from vit_pytorch import ViT as LucidrainViTBase


class LucidrainViT(LucidrainViTBase):
    def __init__(self, learning_tasks, *args, **kwargs):
        LucidrainViTBase.__init__(self, *args, **kwargs)
        
        self.mlp_heads = nn.ModuleDict({})
        self.mlp_head = nn.Identity() # Overwrite the original self.mlp_head in the ViT. Use nn.Identity instead of None just incase some pytorch logic wants modules at not None.

        self.learning_task_list = learning_tasks

        for learning_task in self.learning_task_list:
            self.mlp_heads.update({learning_task.name: nn.Linear(kwargs['dim'], learning_task.output_layer_shape)})

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        outputs = {}
        for task_name, head in self.mlp_heads.items():
            outputs[task_name] = head(x)

        return outputs