import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4


class EFFB4(nn.Module):
    def __init__(self):
        super().__init__()
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        eff_b4 = efficientnet_b4(pretrained=True)
        self.seq=nn.Sequential(*list(eff_b4.children())[:-1])
        
        self.linear0=nn.Linear(in_features=1792, out_features=128, bias=True).to(self.device)
    

    def forward(self, x):
        x=self.seq(x).squeeze(-1).squeeze(-1)
        embedding=self.linear0(x)
        embedding=embedding/embedding.norm(dim=0)
        return embedding
