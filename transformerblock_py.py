
GPT_CONFIG_124={
    "vocab_size":50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":2,
    "drop_rate":0.1,
    "qkv_bias":False

}

import torch.nn as nn
import torch

class LayerNorm(nn.Module):
  def __init__(self,emb_dim):
    super().__init__()
    self.eps=1e-5
    self.scale=nn.Parameter(torch.ones(emb_dim))
    self.shift=nn.Parameter(torch.zeros(emb_dim))
  def forward(self,x):
    mean=x.mean(dim=-1,keepdim=True)
    var=x.var(dim=-1,keepdim=True,unbiased=False)
    norm_x=(x-mean)/torch.sqrt(var+self.eps)
    return self.scale*norm_x+self.shift
class GELU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))
class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers=nn.Sequential(
        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),#expansion
        GELU(),#activation
        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])#contraction
    )
  def forward(self,x):
    return self.layers(x)

class MultiHeadAttention(nn.Module):
  def __init__(self,din,dout,context_length,dropout,num_heads,qkv_bias=False):
    super().__init__()
    assert (dout%num_heads==0)
    self.dout=dout
    self.num_heads=num_heads
    self.head_dim=dout//num_heads

    self.w_query=nn.Linear(din,dout,bias=qkv_bias)
    self.w_key=nn.Linear(din,dout,bias=qkv_bias)
    self.w_value=nn.Linear(din,dout,bias=qkv_bias)

    self.out_proj=nn.Linear(dout,dout)
    self.dropout=nn.Dropout(dropout)
    self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
  def forward(self,x):
    b,num_tokens,din=x.shape

    key=self.w_key(x)
    value=self.w_value(x)
    query=self.w_query(x)

    key=key.view(b,num_tokens,self.num_heads,self.head_dim)
    value=value.view(b,num_tokens,self.num_heads,self.head_dim)
    query=query.view(b,num_tokens,self.num_heads,self.head_dim)

    key=key.transpose(1,2)
    value=value.transpose(1,2)
    query=query.transpose(1,2)

    attn_score=query @ key.transpose(2,3)

    maskbool=self.mask.bool()[:num_tokens,:num_tokens]
    attn_score.masked_fill(maskbool,-torch.inf)

    attn_weight=torch.softmax(attn_score/key.shape[-1]**0.5,dim=-1)
    attn_weight=self.dropout(attn_weight)

    context_vec=(attn_weight @ value).transpose(1,2)

    context_vec=context_vec.contiguous().view(b,num_tokens,self.dout)
    context_vec=self.out_proj(context_vec)
    return context_vec

class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.attn=MultiHeadAttention(
        din=cfg["emb_dim"],
        dout=cfg["emb_dim"],
        context_length=cfg["context_length"],
        dropout=cfg["drop_rate"],
        num_heads=cfg["n_heads"],
        qkv_bias=cfg["qkv_bias"]
    )
    self.ff=FeedForward(cfg)
    self.norm1=LayerNorm(cfg["emb_dim"])
    self.norm2=LayerNorm(cfg["emb_dim"])
    self.drop_shortcut=nn.Dropout(cfg["drop_rate"])
  def forward(self,x):
    shortcut=x
    x=self.norm1(x)
    x=self.attn(x)
    x=self.drop_shortcut(x)
    x=x+shortcut

    shortcut=x
    x=self.norm2(x)
    x=self.ff(x)
    x=self.drop_shortcut(x)
    x=x+shortcut

    return x

torch.manual_seed(123)
x=torch.rand(2,4,768)
block=TransformerBlock(GPT_CONFIG_124)
output=block(x)
print(x.shape)
print(output.shape)
print(output)

"""# **GPT_Model**"""

class GPTModel(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.tok_Embedding=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
    self.pos_Embedding=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
    self.drop_out=nn.Dropout(cfg["drop_rate"])
    self.transformerBlock=nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    ) # create sequential stack of transformerBlock objects .the number is based on n_layers
    self.finalNorm=LayerNorm(cfg["emb_dim"])
    self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
  def forward(self,input):
    print(input.shape)
    b,num_tokens=input.shape
    tok_Embedding=self.tok_Embedding(input)
    pos_Embedding=self.pos_Embedding(torch.arange(num_tokens,device=input.device))   #create a tensor with 0-(tok_length-1) and that tensor is created at same cpu or gpu
    x=tok_Embedding + pos_Embedding
    x=self.drop_out(x)
    x=self.transformerBlock(x)
    x=self.finalNorm(x)
    logits=self.out_head(x)
    return logits

import tiktoken
tokenizer=tiktoken.get_encoding("gpt2")
batch=[]
text1="My name is Manuel"
text2="I have a mobile"
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch=torch.stack(batch,dim=0)  # row based stack
print(batch)

torch.manual_seed(123)  # getting random numbers or weight in each running
model=GPTModel(GPT_CONFIG_124)
logits=model(batch)
print(logits.shape)

#Without unSqueeze
x=torch.tensor([1,23,4])
print(x.shape)
#With Unsqueeze project to 2d
x=x.unsqueeze(0)
print(x.shape)

def generateText(model,input,max_no_token,context_size):
  for _ in range(max_no_token):
    input_s=input[:,-context_size:]  #create inpput-output target pair using context_size
    with torch.no_grad():
      logits=model(input)
    #logits is in dimm3={batch_size,num_token,vocab_size}->{batch_size,vocab_size}
    logits=logits[:,-1,:]
    print(logits.shape)
    #apply softmax to make sum of elements =1
    prob=torch.softmax(logits,dim=-1)
    #highest probability word is taken as next token
    next_tok=torch.argmax(prob,dim=-1,keepdim=True)#keepdim makes the result row wise[[0.1],[0.2]].if keepdim=False then result =[0.1,0.2]
    input=torch.cat((input,next_tok),dim=1)
  return input

t="pay the bill"
encode=tokenizer.encode(t)
print(encode)
tensor=torch.tensor(encode).unsqueeze(0)
print(tensor)

"""If you don’t call **model.eval()** before inference:

Dropout will still randomly remove neurons → inconsistent predictions.

BatchNorm will use batch statistics → unstable outputs if your batch size is small
"""

model.eval()
output=generateText(
  model=model,
  input=tensor,
  max_no_token=10,
  context_size=GPT_CONFIG_124["context_length"]
)
print(output)

final_text=tokenizer.decode(output.squeeze(0).tolist())
print(final_text)