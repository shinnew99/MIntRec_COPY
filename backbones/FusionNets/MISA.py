import torch
from torch import nn
from torch.autograd import Function
from ..SubNets.FeatureNets import BERTEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['MISA']

class ReverseLatyerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    
    @staticmethod
    def forward(ctx, x, p):  #
        ctx.p = p
        
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):   
        output = grad_output.neg() * ctx.p    #grad_output이 여기서 뭐지..
        return output, None
    
    
class MISA(nn.Module):
    
    def __init__(self, args):
        super(MISA, self).__init__()  #여기에서 super은 7번째 줄에 MISA를 의미하나..?
        
        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)
        self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim
        
        self.dropout_rate = args.dropout_rate
        self.output_dim = args.num_labels
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.args = args
        
        rnn = nn.LSTM if args.rnncell == "lstm" else nn.GRU
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0], out_features=args.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNor(args.hidden_size))
        
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=args.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNor(args.hidden_size))
        
        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=args.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNor(args.hidden_size))
        
        
        
        ##############################################
        # private encoders
        ##############################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        # 왜 a_1이 아니고 a_3인지 잘 모르겠음
        
        
        
                
        