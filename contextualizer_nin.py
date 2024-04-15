import torch
from torch import nn





class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


   






class ContextualizerBlock(nn.Module):
    def __init__(self, d_model,d_ffn,dropout,num_tokens):
        super().__init__()
        
        self.context_proj = nn.Linear(d_model,d_model)
        self.mlp = FeedForward(d_model,d_ffn,dropout)     
        self.norm = nn.LayerNorm(d_model)       
        self.upsample = nn.Upsample(scale_factor=num_tokens,mode='nearest')
        self.downsample = nn.Upsample(scale_factor= 1/num_tokens, mode='nearest')
    def forward(self, x):
        res = x
        x = self.norm(x)
        
        context = x
        dim0 = context.shape[0]
        dim1 = context.shape[1]
        dim2 = context.shape[2]
        context = context.reshape([dim0,1,dim1*dim2])
        
        context = self.downsample(context)
        context = context.reshape([dim0,dim2])
        context = self.context_proj(context)
        
        context = context.reshape([dim0,1,dim2])
        context = self.upsample(context)
        context = context.reshape([dim0,dim1,dim2]) 
        x = context
        x = x + res  
        res = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + res
        return out 
        return
 
 
class MixerGatingUnit(nn.Module):
    def __init__(self,d_model,d_ffn,dropout,num_tokens):
        super().__init__()     
        self.Mixer = ContextualizerBlock(d_model,d_ffn,dropout,num_tokens)
        self.proj = nn.Linear(d_model,d_model)

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)  
        v = self.Mixer(v)
        out = u * v
        return out


class ContextualizerNiNBlock(nn.Module):
    def __init__(self, d_model,d_ffn,dropout,num_tokens):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mgu = MixerGatingUnit(d_model,d_ffn,dropout,num_tokens)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out
    
class ContextualizerNiN(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers,dropout,num_tokens):
        super().__init__()
        
        self.model = nn.Sequential(
            
            *[ContextualizerNiNBlock(d_model,d_ffn,dropout,num_tokens) for _ in range(num_layers)],
            
            
        )

    def forward(self, x):
        
        x = self.model(x)
        
        return x
       
       
       
       
       
       
       
       






