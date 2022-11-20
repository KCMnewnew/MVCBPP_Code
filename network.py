import torch
import torch.nn as nn
import torch.nn.functional as F

class Squareplus(nn.Module):
    def __init__(self, b=1.52):
        super(Squareplus, self).__init__()
        self.b = b

    def forward(self, x):
        x = 0.5 * (x + torch.sqrt(x+self.b))
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout):  # 不使用dropout
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, enc_hid_dim, num_layers, bidirectional=True)  # 1,64,10
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):  # [1200,1,3]
        #print('Encoder:-------------------------------------')
        src = src.to(torch.float32)
        src = src.permute(2, 0, 1)  # [3,1200,1]
        enc_output, (s_all, c_all) = self.rnn(src)  # if h_0 is not give, it will be set 0 acquiescently
        #print('enc_output:', enc_output.shape, 's_all:', s_all.shape, 'c_all:', c_all.shape)
        s = torch.tanh(self.fc(torch.cat((s_all[-2, :, :], s_all[-1, :, :]), dim=1)))
        #print('s:', s.shape, 's_all -2,:,: :', s_all[-2, :, :].shape)
        return enc_output, s, c_all

class Encoder_conv(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout): # 不使用dropout
        super(Encoder_conv, self).__init__()
        self.conv = nn.Conv1d(input_dim, enc_hid_dim, input_dim) # 1,64,1
        self.rnn = nn.LSTM(enc_hid_dim, enc_hid_dim, num_layers, bidirectional=True) # 64,64,10
        self.fc = nn.Linear(enc_hid_dim*2 , dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        src =src.to(torch.float32) # [1200,1,3]
        #print('Encoder:-------------------------------------')
        src = self.conv(src)    #  [1200,64,3]
        #print('conv1D->src:',src.shape)
        src = src.permute(2,0,1)# [3,1200,64]
        #print('src:',src.shape)
        enc_output, (h_all,c_all) = self.rnn(src)  #enc_output:[3,1200,128],h_all:[2, 1200, 64],c_all:[2, 1200, 64]
        #print('相等否？:',enc_output[0,:,64:128]==h_all[-1],'c_all:',c_all.shape)
        s_cat_=torch.cat((h_all[-2,:,:], h_all[-1,:,:]), dim = 1) #s_cat_:[1200,128]
        #print('s_cat_:',s_cat_.shape)
        #print('h_all[-2,:,:]:',h_all[-2,:,:],'h_all[-1,:,:]',h_all[-1,:,:])
        s = torch.tanh(self.fc(s_cat_))  #[1200,64]
        #print('s:',s.shape,'h_all -2,:,: :',h_all[-2,:,:].shape)
        return enc_output, s, c_all #[3,1200,128],[1200,64],[2,1200,64]

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False) # (192,64)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)  # 64,1
    def forward(self, s, enc_output):
        #print('Attention:-------------------------------------')
        batch_size = enc_output.shape[1] # 1200
        src_len = enc_output.shape[0]    # 3
        s = s.unsqueeze(1).repeat(1, src_len, 1)  # 【1200 ,3, 64】
        enc_output = enc_output.transpose(0, 1)   # 【1200, 3, 128】
        attn_cat_=torch.cat((s, enc_output), dim = 2) #[1200,3,192]
        energy = torch.tanh(self.attn(attn_cat_)) #[1200,3,64]
        attention = self.v(energy).squeeze(2) #[1200,3]
        return_=F.softmax(attention, dim=1) #[1200,3]
        return return_# [batch_size, seq_len]: 1200, 3


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,1
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.dropout = nn.Dropout(dropout)
        self.f = nn.Softplus()

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128

        a = self.attention(s, enc_output).unsqueeze(1)
        #print('a:', a.shape)
        #print('Decoder:-------------------------------------')
        enc_output = enc_output.transpose(0, 1)  # [1200,3,128]
        #print('enc_output:', enc_output.shape)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)  # [1,1200,128]
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 2, 1200 , 64
        #print('s:', s.shape, 'c_all:', c_all.shape, 'rnn_input:', rnn_input.shape)
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))  # [1,1200,128]
        #print('dec_output:', dec_output.shape)
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        #print('pred:', pred.shape)  # [1200,1]
        return pred, dec_c.squeeze(0)


class Decoder_(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder_, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,1
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.dropout = nn.Dropout(dropout)
        self.f = nn.Sigmoid()

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128
        a = self.attention(s, enc_output).unsqueeze(1)
        enc_output = enc_output.transpose(0, 1)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 2, 1200 , 64
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        return pred, dec_c.squeeze(0)


class Decoder__(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder__, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,1
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.dropout = nn.Dropout(dropout)
        self.f = Squareplus()

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128
        a = self.attention(s, enc_output).unsqueeze(1)
        enc_output = enc_output.transpose(0, 1)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 2, 1200 , 64
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        return pred, dec_c.squeeze(0)


class Decoder_conv(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder_conv, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,10
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.f = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128
        a = self.attention(s, enc_output).unsqueeze(1)
        # enc_output = [batch_size, src_len, enc_hid_dim * 2] 1200,3,128
        enc_output = enc_output.transpose(0, 1)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 20, 1200 , 128
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))
        # 进行自注意力计算
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        return pred, dec_c.squeeze(0)


class Decoder_conv_(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder_conv_, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,10
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.f = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128
        a = self.attention(s, enc_output).unsqueeze(1)
        #print('a:', a.shape)
        #print('Decoder:-------------------------------------')
        enc_output = enc_output.transpose(0, 1)  # [1200,3,128]
        #print('enc_output:', enc_output.shape)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)  # [1,1200,128]
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 20, 1200 , 128
        #print('s:', s.shape, 'c_all:', c_apll.shape, 'rnn_input:', rnn_input.shape)
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))  # [1,1200,128]
        #print('dec_output:', dec_output.shape)
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        #print('pred:', pred.shape)  # [1200,1]
        return pred, dec_c.squeeze(0)


class Decoder_conv__(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super(Decoder_conv__, self).__init__()
        self.output_dim = output_dim  # 1
        self.attention = attention  # (64, 64)
        self.rnn = nn.LSTM(enc_hid_dim * 2, dec_hid_dim, num_layers, bidirectional=True)  # 128,64,10
        self.fc_out = nn.Linear(enc_hid_dim * 2, output_dim)  # 128, 1
        self.f = Squareplus()
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, c_all, enc_output):
        num_layer_bidirectional = c_all.shape[0]  # 2
        dec_input = enc_output  # dec_input = [batch_size, src_length,  dim *2] # 3, 1200, 128
        a = self.attention(s, enc_output).unsqueeze(1)
        enc_output = enc_output.transpose(0, 1)
        rnn_input = torch.bmm(a, enc_output).transpose(0, 1)
        s = s.unsqueeze(0).repeat(num_layer_bidirectional, 1, 1)  # 20, 1200 , 128
        dec_output, (dec_h, dec_c) = self.rnn(rnn_input, (s, c_all))
        dec_output = dec_output.squeeze(0)
        pred = self.fc_out(dec_output)
        pred = self.f(pred)
        return pred, dec_c.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        torch.seed()
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s, s_all = self.encoder(src)
        dec_output, s = self.decoder(s, s_all, enc_output)
        dec_index = torch.argsort(dec_output, dim=0)  # 从小到大的顺序，序列的值，如：56,88,9,1...
        return dec_output, dec_index

class ValueNet(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(ValueNet,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
    def forward(self,inputs):
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x

class Policy_net(nn.Module):
    def __init__(self):
        super(Policy_net, self).__init__()
        INPUT_DIM = 1
        # ENC_EMB_DIM = 64
        # DEC_EMB_DIM = 64
        ENC_HID_DIM = 64
        DEC_HID_DIM = 64
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        NUM_LAYERS = 1
        OUTPUT_DIM = 1
        SEQ_LEN = 3
        DROPOUT = 0.5
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)  # 双向注意力 （64， 64）
        # self_attn = Self_Attention(ENC_HID_DIM, DEC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT)  # (1,64,64,1,0.5)
        enc_conv = Encoder_conv(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT)  # (1,64,64,1,0.5)
        dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT, attn)
        dec_ = Decoder_(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT, attn)
        model0 = Seq2Seq(enc, dec)
        model1 = Seq2Seq(enc, dec_)
        model2 = Seq2Seq(enc_conv, dec_)
        model3 = Seq2Seq(enc_conv, dec)
        self.policy_net = model0
        #self.v_net = ValueNet()

