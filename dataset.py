import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    ### how is this init diff from other parameters in other tasks? why we dont have embeddings here
    def __init__(self, ds,tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        ### do we need to initialize sos tokens here ?
        ## what is this here ??
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
     

    
    def __getitem__(self, index):

    ### what are all we doing in this in get item and what are we doing ? can this be done in separate method? 
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        ### lets do padding

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #( ### because of SOS and EOS)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 ## (we only add SOS, but y ??)

        ### padding tokens should never become negetive

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long') ### whatt ??? how ??
        
        ## Adding SOS and EOS to Source Text
        encoder_input = torch.cat(
            [
            self.sos_token, 
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        ## Adding SOS to Target Text

        decoder_input = torch.cat(

            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        ## addig EOS token what is label for ? 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
            
        )

### what are these checking ? 
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            'encoder_input': encoder_input, #(seq_len)
            'decoder_input': decoder_input,

            ## dont want these tokens to be seen by the self attention
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            ### we need causal mask for decoder
            'decoder_mask': ((decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()) & (causal_mask(decoder_input.size(0))), ##(1,seq_len) & (1,seq_len, seq_len)
            'label' : label,
            'src_text': src_text,
            'tgt_text': tgt_text

        }
### check this for smaller inputs or own inputs
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  