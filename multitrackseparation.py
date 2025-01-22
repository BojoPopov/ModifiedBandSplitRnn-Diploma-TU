import typing as tp
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import torchaudio.transforms as T
import torch.nn as nn
import torch
import pytorch_lightning as pl

def initialize_weights_with_normal(module, mean=0.0, std=0.1):
    """
    Apply Normal (Gaussian) initialization to weights of applicable layers in a PyTorch model.
    Zeroes out biases if they exist.

    Args:
        module: A PyTorch module to initialize.
        mean: Mean for normal initialization (default: 0.0).
        std: Standard deviation for normal initialization (default: 1.0).
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.zeros_(param)

    elif isinstance(module, nn.MultiheadAttention):
        nn.init.normal_(module.in_proj_weight, mean=mean, std=std)
        nn.init.normal_(module.out_proj.weight, mean=mean, std=std)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
        pass  # No initialization required for pooling layers.

def initialize_weights_with_xavier(module):
    """
    Apply Xavier initialization to weights of applicable layers in a PyTorch model.
    Zeroes out biases if they exist.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:  # Initialize weights
                nn.init.xavier_uniform_(param)
            elif "bias" in name:  # Zero-out biases
                nn.init.zeros_(param)

    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)  # Initialize scale parameters
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # Initialize shift parameters

    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
        # Pooling layers do not have weights or biases, so no initialization needed
        pass
def initialize_weights_with_uniform(module, low=-0.1, high=0.1):
    """
    Apply uniform initialization to weights of applicable layers in a PyTorch model.
    Zeroes out biases if they exist.
    Parameters:
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.uniform_(module.weight, a=low, b=high)  # Uniform for Conv layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=low, b=high)  # Uniform for FC layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:  # Initialize weights
                nn.init.uniform_(param, a=low, b=high)
            elif "bias" in name:  # Zero-out biases
                nn.init.zeros_(param)

    elif isinstance(module, nn.MultiheadAttention):
        nn.init.uniform_(module.in_proj_weight, a=low, b=high)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        nn.init.uniform_(module.out_proj.weight, a=low, b=high)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)  # Initialize scale parameters
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # Initialize shift parameters

    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
        pass  # Pooling layers don't have weights or biases
def initialize_weights_with_kaiming(module):
    """
    Apply Kaiming initialization to weights of applicable layers in a PyTorch model.
    Zeroes out biases if they exist.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(module.weight, a=0, nonlinearity='relu')  # Kaiming uniform for Conv layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=0, nonlinearity='relu')  # Kaiming uniform for FC layers
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:  # Initialize weights
                nn.init.kaiming_uniform_(param, a=0, nonlinearity='relu')
            elif "bias" in name:  # Zero-out biases
                nn.init.zeros_(param)

    elif isinstance(module, nn.MultiheadAttention):
        nn.init.kaiming_uniform_(module.in_proj_weight, a=0, nonlinearity='relu')
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        nn.init.kaiming_uniform_(module.out_proj.weight, a=0, nonlinearity='relu')
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)  # Initialize scale parameters
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # Initialize shift parameters

    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
        pass  # Pooling layers don't have weights or biases



def get_fftfreq(sr: int = 44100,n_fft: int = 2048) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out
def get_subband_indices(freqs: torch.Tensor,splits: tp.List[tp.Tuple[int, int]],) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices
def freq2bands(bandsplits: tp.List[tp.Tuple[int, int]],sr: int = 44100,n_fft: int = 2048) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices


def select_activation(activation_type: str) -> nn.modules.activation:
            if activation_type == 'none':
                return nn.Identity()
            elif activation_type == 'relu':
                return nn.ReLU()
            elif activation_type == 'tanh':
                return nn.Tanh()
            elif activation_type == 'gelu':
                return nn.GELU()
            elif activation_type == 'softmax':
                return nn.Softmax(dim=-1)
            elif activation_type == 'prelu':
                return nn.PReLU()
            elif activation_type == 'sigmoid':
                return nn.Sigmoid()
            elif activation_type == 'leakyrelu':
                return nn.LeakyReLU()
            elif activation_type == 'hardswish':
                return nn.Hardswish()
            elif activation_type == 'mish':
                return nn.Mish()
            elif activation_type == 'silu':
                return nn.SiLU()
            elif activation_type == 'celu':
                return nn.CELU()
            elif activation_type == 'elu':
                return nn.ELU()
            elif activation_type == 'selu':
                return nn.SELU()
            elif activation_type == 'rrelu':
                return nn.RReLU()
            elif activation_type == 'relu6':
                return nn.ReLU6()
            elif activation_type == 'hardshrink':
                return nn.Hardshrink()
            elif activation_type == 'hardsigmoid':
                return nn.Hardsigmoid()
            elif activation_type == 'hardtanh':
                return nn.Hardtanh()
            elif activation_type == 'logsigmoid':
                return nn.LogSigmoid()
            elif activation_type == 'logsoftmax':
                return nn.LogSoftmax(dim=-1)
            elif activation_type == 'softmax2d':
                return nn.Softmax2d()
            elif activation_type == 'softmin':
                return nn.Softmin(dim=-1)
            elif activation_type == 'softplus':
                return nn.Softplus()
            elif activation_type == 'softshrink':
                return nn.Softshrink()
            elif activation_type == 'softsign':
                return nn.Softsign()
            elif activation_type == 'tanhshrink':
                return nn.Tanhshrink()
            else:
                print("select activation wrong input")

def select_norm(activation_type: str) -> nn.modules:
        if activation_type == 'none':
            return nn.Identity
        elif activation_type == 'batchnorm1d':
            return nn.BatchNorm1d
        elif activation_type == 'batchnorm2d':
            return nn.BatchNorm2d
        elif activation_type == 'groupnorm':
            return nn.GroupNorm
        elif activation_type == 'instancenorm1d':
            return nn.InstanceNorm1d
        elif activation_type == 'instancenorm2d':
            return nn.InstanceNorm2d
        elif activation_type == 'layernorm':
            return nn.LayerNorm
        elif activation_type == 'rmsnorm':
            return nn.RMSNorm
        else:
            print("select norm wrong input")



class CAM(nn.Module):
    def __init__(self, ch,output,mlp_dim,act_mlp,act_glu, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.transform = nn.Conv1d(ch,output,kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mlp_dim),
            select_activation(act_mlp),
            nn.Linear(mlp_dim, output)
        )

        self.sigmoid = GLU(output, act_glu)

    def forward(self, x):

        x1 = self.avg_pool(x).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1)
        refined_feats = self.transform(x) * feats

        return refined_feats
class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats
class CBAM(nn.Module):
    def __init__(self, channel, output,mlp_dim,act_mlp,act_glu):
        super().__init__()

        self.ca = CAM(channel, output,mlp_dim,act_mlp,act_glu)
        self.sa = SAM()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class GLU(nn.Module):
    """
    GLU Activation Module.
    """
    def __init__(self, input_dim: int, activation):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = select_activation(activation)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x[..., :self.input_dim] * self.sigmoid(x[..., self.input_dim:])
        return x
class MLP(nn.Module):
    """
    Just a simple MLP with tanh activation (by default).
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: str = 'tanh',
            activation_glu: str = 'sigmoid'
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            select_activation(activation),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim, activation_glu)
        )




    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(self,
                 sr: int,
                 n_fft: int,
                 bandsplits: tp.List[tp.Tuple[int, int]],
                 act1:str,
                 act2:str,
                 act3:str,
                 act4:str,
                 act5:str,
                 act6:str,
                 act7:str,
                 act8:str,
                 act9:str,
                 act10:str,
                 act_mlp:str,
                 act_glu_mlp:str,
                 act_cbam:str,
                 act_glu_cbam:str,
                 norm1:str,
                 norm2:str,
                 norm3:str,
                 norm4:str,
                 norm5:str,
                 t_timesteps: int = 517,
                 fc_dim: int = 128,
                 mlp_dim: int = 512,
                 complex_as_channel: bool = True,
                 is_mono: bool = False):

        super(BandSplitModule, self).__init__()



        frequency_mul = 4
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2
        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)



        "Before Split"
        self.act1 = select_activation(act1)

        if norm1 == 'layernorm' or norm1 == 'rmsnorm':
            self.norm1 = select_norm(norm1)([frequency_mul, n_fft // 2 + 1, t_timesteps])
        elif norm1== 'groupnorm':
            self.norm1 = select_norm(norm1)(frequency_mul, frequency_mul)
        else:
            self.norm1 = select_norm(norm1)(frequency_mul)

        self.act2 = select_activation(act2)



        "After Split Before FC"
        self.act3 = select_activation(act3)

        if norm2 == 'layernorm' or norm2== 'rmsnorm':
            self.norm2 = nn.ModuleList([
                select_norm(norm2)([(e - s) * frequency_mul, t_timesteps])
                for s, e in self.bandwidth_indices
            ])
        elif norm2== 'groupnorm':
            self.norm2 = nn.ModuleList([
                select_norm(norm2)((e - s) * frequency_mul, (e - s) * frequency_mul)
                for s, e in self.bandwidth_indices
            ])
        else:
            self.norm2 = nn.ModuleList([
                select_norm(norm2)((e - s) * frequency_mul)
                for s, e in self.bandwidth_indices
            ])

        self.act4 = select_activation(act4)



        "After Split After FC"
        self.act5 = select_activation(act5)

        if norm3 == 'layernorm' or norm3 == 'rmsnorm':
            self.norm3 = nn.ModuleList([
                select_norm(norm3)([t_timesteps, fc_dim])
                for s, e in self.bandwidth_indices
            ])
        elif norm3 == 'groupnorm':
            self.norm3 = nn.ModuleList([
                select_norm(norm3)(t_timesteps, t_timesteps)
                for s, e in self.bandwidth_indices
            ])
        else:
            self.norm3 = nn.ModuleList([
                select_norm(norm3)(t_timesteps)
                for s, e in self.bandwidth_indices
            ])

        self.act6 = select_activation(act6)



        "After Split After CBAM"
        self.act7 = select_activation(act7)

        if norm4 == 'layernorm'or norm4== 'rmsnorm':
            self.norm4 = nn.ModuleList([
                select_norm(norm4)([t_timesteps, fc_dim])
                for s, e in self.bandwidth_indices
            ])
        elif norm4== 'groupnorm':
            self.norm4 = nn.ModuleList([
                select_norm(norm4)(t_timesteps, t_timesteps)
                for s, e in self.bandwidth_indices
            ])
        else:
            self.norm4 = nn.ModuleList([
                select_norm(norm4)(t_timesteps)
                for s, e in self.bandwidth_indices
            ])

        self.act8 = select_activation(act8)



        "After Merge"
        self.act9 = select_activation(act9)

        if norm5 == 'layernorm' or norm5== 'rmsnorm':
            self.norm5 = select_norm(norm5)([41, t_timesteps, fc_dim])
        elif norm5== 'groupnorm':
            self.norm5 = select_norm(norm5)(41, 41)
        else:
            self.norm5 = select_norm(norm5)(41)

        self.act10 = select_activation(act10)



        "Modules"
        self.cbam = nn.ModuleList([CBAM(fc_dim, fc_dim, mlp_dim, act_cbam, act_glu_cbam) for s, e in self.bandwidth_indices ])

        self.fcs = nn.ModuleList([
            MLP((e - s) * frequency_mul, mlp_dim, fc_dim, act_mlp, act_glu_mlp)
            for s, e in self.bandwidth_indices
        ])

        # self.mha = nn.ModuleList([
        #     nn.MultiheadAttention(337,337,batch_first=True)
        #     for s, e in self.bandwidth_indices
        # ])



    def generate_subband(
            self,
            x: torch.Tensor
    ) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]



    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []



        "Before Split"
        B, C, F, T = x.shape
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).contiguous()
        x = x.reshape(B, -1,F, T).contiguous()
        x = self.act1(x)
        x = self.norm1(x)
        x = self.act2(x)
        x = x.reshape(B, C, 2, F, T).contiguous()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = torch.view_as_complex(x)



        "After Split"
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape

            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x)
                x = x.permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T)
            # run through model



            "Before FC"
            x=self.act3(x)
            x = self.norm2[i](x)
            x=self.act4(x)



            "FC"
            x = x.transpose(-1, -2)
            x = self.fcs[i](x)



            "After FC"
            x = self.act5(x)
            x = self.norm3[i](x)
            x = self.act6(x)


            "CBAM"
            x = x.transpose(-1, -2)
            x = self.cbam[i](x)
            x = x.transpose(-1, -2)



            "After CBAM"
            x=self.act7(x)
            x = self.norm4[i](x)
            x=self.act8(x)
            xs.append(x)



        "After Merge"
        x = torch.stack(xs, dim=1)
        x=self.act9(x)
        x = self.norm5(x)
        x= self.act10(x)



        return x



class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            mlp_dim: int,
            activation: str,
            activation_glu: str,
            activation_cbam: str,
            activation_glu_cbam: str,
            rnn_type: str = 'lstm',
            bidirectional: bool = True
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = MLP(hidden_dim_size * 2 if bidirectional else hidden_dim_size, hidden_dim_size, input_dim_size, activation=activation, activation_glu=activation_glu)
        self.cbam = CBAM(input_dim_size, input_dim_size, mlp_dim, activation_cbam, activation_glu_cbam)

    def forward(self,x: torch.Tensor):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)
        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]
        out = self.groupnorm(
            out.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]
        out = out.transpose(-1,-2)
        out = self.cbam(out)
        out = out.transpose(-1,-2)
        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x
class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            mlp_dim: int,
            activation:str,
            activation_glu:str,
            activation_cbam: str,
            activation_glu_cbam: str,
            rnn_type: str = 'lstm',
            bidirectional: bool = True,
            num_layers: int = 12,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])

        for _ in range(num_layers):
            rnn_across_t = RNNModule(
                input_dim_size, hidden_dim_size,mlp_dim, activation,activation_glu,activation_cbam,activation_glu_cbam, rnn_type, bidirectional,
            )
            rnn_across_k = RNNModule(
                input_dim_size, hidden_dim_size,mlp_dim, activation,activation_glu,activation_cbam,activation_glu_cbam, rnn_type, bidirectional,
            )
            self.bsrnn.append(
                nn.Sequential(rnn_across_t, rnn_across_k)
            )

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)
        return x




class MaskEstimationModule(nn.Module):
    """
    MaskEstimation (3rd) Module of BandSplitRNN.
    Recreates from input initial subband dimensionality via running through LayerNorms+MLPs and forms the T-F mask.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            act1: str,
            act2: str,
            act3: str,
            act4: str,
            act5: str,
            act6: str,
            act7: str,
            act8: str,
            act9: str,
            act10: str,
            act_mlp: str,
            act_glu_mlp: str,
            act_cbam: str,
            act_glu_cbam: str,
            norm1: str,
            norm2: str,
            norm3: str,
            norm4: str,
            norm5: str,
            t_timesteps: int = 517,
            fc_dim: int = 128,
            mlp_dim: int = 512,
            complex_as_channel: bool = True,
            is_mono: bool = False,
    ):
        super(MaskEstimationModule, self).__init__()
        frequency_mul = 4
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.frequency_mul = frequency_mul
        self.bandwidths = [(e - s) for s, e in freq2bands(bandsplits, sr, n_fft)]



        "Before Split"
        self.act1 = select_activation(act1)

        if norm1 == 'layernorm' or norm1== 'rmsnorm':
            self.norm1 = select_norm(norm1)([41, t_timesteps, fc_dim])
        elif norm1== 'groupnorm':
            self.norm1 = select_norm(norm1)(41, 41)
        else:
            self.norm1 = select_norm(norm1)(41)

        self.act2 = select_activation(act2)



        "After Split Before FC"
        self.act3 = select_activation(act3)

        if norm2 == 'layernorm'or norm2== 'rmsnorm':
            self.norm2 = nn.ModuleList([
                select_norm(norm2)([t_timesteps, fc_dim])
                for bw in self.bandwidths
            ])
        elif norm2== 'groupnorm':
            self.norm2 = nn.ModuleList([
                select_norm(norm2)(t_timesteps, t_timesteps)
                for bw in self.bandwidths
            ])
        else:
            self.norm2 = nn.ModuleList([
                select_norm(norm2)(t_timesteps)
                for bw in self.bandwidths
            ])

        self.act4 = select_activation(act4)



        "After Split After FC"
        self.act5 = select_activation(act5)

        if norm3 == 'layernorm' or norm3== 'rmsnorm':
            self.norm3 = nn.ModuleList([
                select_norm(norm3)([t_timesteps,bw * frequency_mul])
                for bw in self.bandwidths            ])
        elif norm3== 'groupnorm':
            self.norm3 = nn.ModuleList([
                select_norm(norm3)(t_timesteps,t_timesteps)
                for bw in self.bandwidths            ])
        else:
            self.norm3 = nn.ModuleList([
                select_norm(norm3)(t_timesteps)
                for bw in self.bandwidths            ])

        self.act6 = select_activation(act6)



        "After Split After CBAM"
        self.act7 = select_activation(act7)

        if norm4 == 'layernorm' or norm4== 'rmsnorm':
            self.norm4 = nn.ModuleList([
                select_norm(norm4)([t_timesteps,bw * frequency_mul])
                for bw in self.bandwidths            ])
        elif norm4== 'groupnorm':
            self.norm4 = nn.ModuleList([
                select_norm(norm4)(t_timesteps,t_timesteps)
                for bw in self.bandwidths            ])
        else:
            self.norm4 = nn.ModuleList([
                select_norm(norm4)(t_timesteps)
                for bw in self.bandwidths            ])

        self.act8 = select_activation(act8)




        "After Merge"
        self.act9 = select_activation(act9)

        if norm5 == 'layernorm' or norm5 == 'rmsnorm':
            self.norm5 = select_norm(norm5)([4, n_fft // 2 + 1, t_timesteps])
        elif norm5 == 'groupnorm':
            self.norm5 = select_norm(norm5)(4, 4)
        else:
            self.norm5 = select_norm(norm5)(4)

        self.act10 = select_activation(act10)



        "Modules"
        self.cbam = nn.ModuleList([CBAM(bw * frequency_mul,bw * frequency_mul,mlp_dim, act_mlp=act_cbam, act_glu=act_glu_cbam) for bw in self.bandwidths])

        self.mlp = nn.ModuleList([
            MLP(fc_dim, mlp_dim, bw * frequency_mul, activation=act_mlp, activation_glu=act_glu_mlp)
            for bw in self.bandwidths
        ])
        # self.mha = nn.ModuleList([
        #     nn.MultiheadAttention(337,337,batch_first=True)
        #     for bw in self.bandwidths
        # ])



    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, k_subbands, time, fc_shape]
        Output: [batch_size, freq, time]
        """
        outs = []
        "Before Split"
        x = self.act1(x)
        x = self.norm1(x)
        x = self.act2(x)



        "Split"
        for i in range(x.shape[1]):
            # run through model
            out = x[:, i]



            "After Split"
            out = self.act3(out)
            out = self.norm2[i](out)
            out = self.act4(out)


            "FC"
            out = self.mlp[i](out)



            "After FC"
            out = self.act5(out)
            out = self.norm3[i](out)
            out = self.act6(out)


            "CBAM"
            out = out.transpose(-1,-2)
            out = self.cbam[i](out)
            out = out.transpose(-1,-2)



            "After CBAM"
            out = self.act7(out)
            out = self.norm4[i](out)
            out = self.act8(out)



            "Merge"
            B, T, F = out.shape
            # return to complex
            if self.cac:
                out = out.permute(0, 2, 1).contiguous()
                out = out.view(B, -1, 2, F//self.frequency_mul, T).permute(0, 1, 3, 4, 2)
                out = torch.view_as_complex(out.contiguous())
            else:
                out = out.view(B, -1, F//self.frequency_mul, T).contiguous()
            outs.append(out)

        # concat all subbands
        outs = torch.cat(outs, dim=-2)



        "After Merge"
        B, C, F, T = outs.shape
        outs = torch.view_as_real(outs).permute(0, 1, 4, 2, 3).contiguous()
        outs = outs.reshape(B, -1,F, T).contiguous()
        outs = self.act9(outs)
        outs = self.norm5(outs)
        outs = self.act10(outs)
        outs = outs.reshape(B, C, 2, F, T).contiguous()
        outs = outs.permute(0, 1, 3, 4, 2).contiguous()
        outs = torch.view_as_complex(outs)



        return outs



class BandSplitRNN(pl.LightningModule):
    """
    BandSplitRNN with learnable window function for STFT.
    """

    def __init__(
            self,
            bandsplits: tp.List[tp.Tuple[int, int]],
            sr: int=44100,
            n_fft: int=2048,
            complex_as_channel: bool=True,
            is_mono: bool=False,
            bottleneck_layer: str='rnn',
            t_timesteps: int=337,
            fc_dim: int=128,
            rnn_dim: int=256,
            rnn_type: str="LSTM",
            bidirectional: bool=True,
            num_layers: int=1,
            mlp_dim: int=512,
            window1='hann',
            window2='hann',
            act1='selu',
            act2='none',
            act3='selu',
            act4='relu',
            act5='gelu',
            act6='none',
            act7='tanh',
            act8='selu',
            act9='relu',
            act10='selu',
            act_mlp='selu',
            act_glu_mlp='sigmoid',
            act_cbam='sigmoid',
            act_glu_cbam='sigmoid',
            norm1='groupnorm', #has to be 2d
            norm2='none', #has to be 1d
            norm3='rmsnorm', #has to be 1d
            norm4='none', #has to be 1d
            norm5='none', #has to be 2d
            return_mask: bool = False,
            device: str ="cuda" if torch.cuda.is_available() else "cpu",
            lr = 1e-4
    ):
        super(BandSplitRNN, self).__init__()
        self.window1 = window1
        self.window2 = window2
        self.device_now = device
        self.lr = lr

        # STFT and iSTFT modules (default window is 'hann')
        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
            act1=act1,
            act2=act2,
            act3=act3,
            act4=act4,
            act5=act5,
            act6=act6,
            act7=act7,
            act8=act8,
            act9=act9,
            act10=act10,
            act_mlp=act_mlp,
            act_glu_mlp=act_glu_mlp,
            act_cbam=act_cbam,
            act_glu_cbam=act_glu_cbam,
            norm1=norm1,
            norm2=norm2,
            norm3=norm3,
            norm4=norm4,
            norm5=norm5,
        )


        self.bandsequence = BandSequenceModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_layers=num_layers,
                activation=act_mlp,
                activation_glu=act_glu_mlp,
                activation_cbam=act_cbam,
                activation_glu_cbam=act_glu_cbam,
                mlp_dim=mlp_dim
            )

        # Ensure that the window name is passed as a string, not a tensor
        self.stft = T.Spectrogram(n_fft=2048, hop_length=525, power=None).to('cpu')
        self.istft = T.InverseSpectrogram(n_fft=2048, hop_length=525).to('cpu')
        # Decoder layer
        self.maskest = MaskEstimationModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
            act1=act1,
            act2=act2,
            act3=act3,
            act4=act4,
            act5=act5,
            act6=act6,
            act7=act7,
            act8=act8,
            act9=act9,
            act10=act10,
            act_mlp=act_mlp,
            act_glu_mlp=act_glu_mlp,
            act_cbam=act_cbam,
            act_glu_cbam=act_glu_cbam,
            norm1=norm1,
            norm2=norm2,
            norm3=norm3,
            norm4=norm4,
            norm5=norm5,
        )
        self.cac = complex_as_channel
        self.return_mask = return_mask

        self.apply(initialize_weights_with_uniform)

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.maskest(x)  # [batch_size, freq, time]

        return x

    def forward(self, x: torch.Tensor):

        #Normalize waveform
        wav_mean = x.mean(dim=(1,2), keepdim=True)
        wav_std = x.std(dim=(1,2), keepdim=True)
        x = (x - wav_mean) / (wav_std + 1e-5)

        #Turn to stft
        x = self.stft(x)

        #Normalize stft
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        #Create 4 copies for each stem and stack
        x = torch.cat([x,x,x,x], dim=1)

        # Compute T-F mask
        mask = self.compute_mask(x)

        #Separate into stems
        bass = mask[:,0:2,:,:]
        drums = mask[:,2:4,:,:]
        other = mask[:,4:6,:,:]
        vocals = mask[:,6:8,:,:]
        # Multiply with original tensor
        bass *= x[:,0:2,:,:]
        drums *= x[:,0:2,:,:]
        other *= x[:,0:2,:,:]
        vocals *= x[:,0:2,:,:]

        # Denormalize
        bass = bass * std + mean
        drums = drums * std + mean
        other = other * std + mean
        vocals = vocals * std + mean

        # Inverse STFT
        bass = self.istft(bass)
        drums =self.istft(drums)
        other = self.istft(other)
        vocals = self.istft(vocals)

        # Denormalize waveform
        bass = bass * wav_std + wav_mean
        drums = drums * wav_std + wav_mean
        other = other * wav_std + wav_mean
        vocals = vocals * wav_std + wav_mean
        return bass, drums, other, vocals
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # scheduler = {
        #     'scheduler': OneCycleLR(optimizer, max_lr=1e-4, epochs=100, steps_per_epoch=2745, three_phase=True),
        #     'interval': 'step',
        #     'frequency': 1,
        #     'name': 'linear_warmup_lr'
        # }

        return [optimizer]

    def training_step(self, batch, batch_idx):
        mix = batch.to(torch.float) # Shape: (B, 4, 2, H, W)
        bass = mix[:, 0:1, :, :].squeeze(1)
        drums = mix[:, 1:2, :, :].squeeze(1)
        other = mix[:, 2:3, :, :].squeeze(1)
        vocals = mix[:, 3:4, :, :].squeeze(1)
        mix = bass+drums+other+vocals
        # Forward pass
        p_bass, p_drums, p_other, p_vocals = self(mix)

        # p_mix = p_bass+p_drums+p_other+p_vocals
        # loss =  F.l1_loss(p_bass,bass) + F.l1_loss(p_drums,drums) + F.l1_loss(p_other,other) + F.l1_loss(p_vocals,vocals) + F.l1_loss(p_mix,mix) + F.l1_loss((mix-p_bass),(mix-bass)) + F.l1_loss((mix-p_drums),(mix-drums)) +F.l1_loss((mix-p_other),(mix-other)) +F.l1_loss((mix-p_vocals),(mix-vocals)) +F.l1_loss((mix-p_mix),torch.zeros_like(mix)) +  F.l1_loss(torch.view_as_real(self.stft(p_bass)),torch.view_as_real(self.stft(bass))) + F.l1_loss(torch.view_as_real(self.stft(p_drums)),torch.view_as_real(self.stft(drums))) + F.l1_loss(torch.view_as_real(self.stft(p_other)),torch.view_as_real(self.stft(other))) + F.l1_loss(torch.view_as_real(self.stft(p_vocals)),torch.view_as_real(self.stft(vocals))) + F.l1_loss(torch.view_as_real(self.stft(p_mix)),torch.view_as_real(self.stft(mix))) +  F.l1_loss(torch.view_as_real(self.stft(mix-p_bass)),torch.view_as_real(self.stft(mix-bass))) + F.l1_loss(torch.view_as_real(self.stft(mix-p_drums)),torch.view_as_real(self.stft(mix-drums))) + F.l1_loss(torch.view_as_real(self.stft(mix-p_other)),torch.view_as_real(self.stft(mix-other))) + F.l1_loss(torch.view_as_real(self.stft(mix-p_vocals)),torch.view_as_real(self.stft(mix-vocals))) + F.l1_loss(torch.view_as_real(self.stft(mix-p_mix)),torch.zeros_like(torch.view_as_real(self.stft(mix))))
        # loss = loss / 10
        loss = F.l1_loss(p_bass, bass) + F.l1_loss(p_drums, drums) + F.l1_loss(p_other, other) + F.l1_loss(p_vocals,vocals) + F.l1_loss(torch.view_as_real(self.stft(p_bass)),torch.view_as_real(self.stft(bass))) + F.l1_loss(torch.view_as_real(self.stft(p_drums)),torch.view_as_real(self.stft(drums))) + F.l1_loss(torch.view_as_real(self.stft(p_other)),torch.view_as_real(self.stft(other))) + F.l1_loss(torch.view_as_real(self.stft(p_vocals)),torch.view_as_real(self.stft(vocals)))
        # Compute loss (Mean Squared Error)

        # Check for NaN or Inf in loss
        if not torch.isfinite(loss):
            self.log("train_loss", float('nan'), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            raise ValueError("NaN or Inf detected in the loss. Stopping training.")

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

if __name__ == '__main__':
    batch_size, n_channels, freq = 4, 2, 176400
    in_features = torch.rand(batch_size, n_channels, freq, dtype=torch.float32)
    splits_v7 = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    model = BandSplitRNN(bandsplits=splits_v7)
    _ = model.eval()

    with torch.no_grad():
        out_features, wav = model(in_features)







