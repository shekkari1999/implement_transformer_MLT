#### First thing we build is input Embedding

import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size): #### giving dimension and vocab_size to build embeddings
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model, seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        #### we need to create a matrix shape of (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        position = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1) #(seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        ## applying sine to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe',pe)

    def forward(self, x):
        pe = self.pe[:, :x.shape[1], :]
        pe.requires_grad = False
        x = x + pe
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) ## last dimension ? 
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #( batch, seq_leng, d_model) --> (batch, seq_len, dff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h,dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)  # why not self. ??
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            ## what is happening here ?
        attention_scores = attention_scores.softmax(dim = -1) ##( batch, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> same
        key = self.w_k(k) # (batch, seq_len, d_model) --> same
        value = self.w_v(v) # (batch, seq_len, d_model) --> same

    ### (batch, seq_len, d_model) --> (batch, seq_len, h, dk) --> (batch, h, seq_len, dk)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        ## what does x and scores mean here? arent they the same ?
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        ## whats happening here ? 
        #(batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch , seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)



class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        ## what is this call ??
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
 

### defining the MEGA encoder Block

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttention, ffb : FFBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = ffb

        ## what the hell is this ? how does this work ?
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x , src_mask): ###why source mask. how do we know when to take this parmeter in ?

        # dafaqq is happening here
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# how is encoder diff from encoder block damn ?
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    ## I have no clue about this forward method though
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        

        
class DecoderBlock(nn.Module):
    ## what is this kind of parameter initialization ? 
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block : FFBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        ## why are we not passing this one in parameters ??
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    ## what is the role of src_mask and tgt_mask ?
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        ## I need to understand the whole flow here
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
    ## Need to understand whole flow of Decoder and whats happening
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):        
        #(batch,seq, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) ### why -1 ? what is there ? 
    

class Transformer(nn.Module):
    ## i guess we missed some layers like norm, we included some like these. why? double check
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    ### what does decode do ?
    def decode(self, encoder_output, src_mask, tgt, tgt_mask): ### why ? 
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    ### what does project do ? 
    def project(self, x):
        return self.proj_layer(x)


##### we need to combine all these. given hyper parameters build the transformer

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_length, tgt_seq_length, d_model = 512,N = 6, h = 8, dropout = 0.1, d_ff = 2048 ):
    
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model ,tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_length, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_length, dropout)

    ## create encoder blocks

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FFBlock(d_model, d_ff, dropout)
        # encoder_block ?? what is this ? 
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    ### create decoder block

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)

        feed_forward_block = FFBlock(d_model, d_ff, dropout)
        # encoder_block ?? what is this ? 
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    ### create the encoder and decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    ## creating the projection layer

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    ### create the transformer

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    ### Initialize parameters

    for p in transformer.parameters():
        ## what is this ? 
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
