import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
import matplotlib.pyplot as plt


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def round_channels(c, divisor=64):
    return int((c + divisor - 1) // divisor) * divisor

## Gated-Dconv Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,LayerNorm_type):
        super(FeedForward, self).__init__()

        hidden_features = round_channels(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.dwconv2 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                groups=hidden_features * 2, bias=bias)

        self.dwconv3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=3,
                                 groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features*3, dim, kernel_size=1, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_features , out_channels=hidden_features , kernel_size=1, padding=0, stride=1, bias=True),
        )
        
        # Simplified Channel Attention
        self.sca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_features //2 , out_channels=hidden_features //2, kernel_size=1, padding=0, stride=1, bias=True),
        )
        
        self.sca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels=hidden_features //2 , out_channels=hidden_features //2, kernel_size=1, padding=0, stride=1, bias=True),
        )
        
        self.norm=LayerNorm(hidden_features * 3,LayerNorm_type)

        self.sg=SimpleGate()

    def forward(self, x):
        h = self.project_in(x)
        x1 = self.dwconv1(h)
        x2 = self.dwconv2(h)
        x3 = self.dwconv3(h)

        x1 = self.sg(x1)
        x2 = self.sg(x2)
        x3 = self.sg(x3)

        x1_avg, x1_max = x1.chunk(2, dim=1)
        x1_avg = self.sca_avg(x1_avg)*x1_avg
        x1_max = self.sca_max(x1_max)*x1_max
        x1 = torch.cat([x1_avg, x1_max], dim=1)
        
        x2_avg, x2_max = x2.chunk(2, dim=1)
        x2_avg = self.sca_avg(x2_avg)*x2_avg
        x2_max = self.sca_max(x2_max)*x2_max
        x2 = torch.cat([x2_avg, x2_max], dim=1)
        
        x3_avg, x3_max = x3.chunk(2, dim=1)
        x3_avg = self.sca_avg(x3_avg)*x3_avg
        x3_max = self.sca_max(x3_max)*x3_max
        x3 = torch.cat([x3_avg, x3_max], dim=1)

        x4 = torch.concat([x1,x2,x3],dim=1)
        x4 = self.norm(x4)
        # x = F.gelu(x1) * x2
        x4 = self.project_out(x4)
        return x4+x

# Multi-DConv Head Transposed Self-Attention (MDTA)
class PA_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias,t_dim = 3):
        super(PA_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.q_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.k_concat = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.v_concat = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, f_out):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        q_T = self.q_dwconv_T(self.q_T(f_out))

        k = self.k_dwconv(self.k(x))
        k_T = self.k_dwconv_T(self.k_T(f_out))

        v = self.v_dwconv(self.v(x))
        v_T = self.v_dwconv_T(self.v_T(f_out))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_T = rearrange(q_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_T = rearrange(k_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_T = rearrange(v_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q_T = torch.nn.functional.normalize(q_T, dim=-1)
        k_T = torch.nn.functional.normalize(k_T, dim=-1)

        attn1 = (q @ k_T.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v_T)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        attn2 = (q_T @ k.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out1 + out2

        # out = self.project_out(out)
        return out
##########################################################################
class TransformerBlock_eca(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_eca, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention_eca(num_heads, 3, bias)
        self.atten = PA_MSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, LayerNorm_type)


    def forward(self, x, time, f_out):
        # x = x + self.atten(self.norm1(x)+ time, f_out)
        x1 = self.norm1(x)
        x1 = self.atten(x1 + time, f_out)
        x = x + x1
        x = x + self.ffn(self.norm2(x) + time)
        # with torch.no_grad():
        #     feature_map = x1[0, 0].cpu().numpy()  # Select first sample, first channel
        #
        #     plt.imshow(feature_map, cmap='viridis')  # You can choose other color maps as well
        #     plt.colorbar()
        #     plt.title('Feature Map of x1')
        #     plt.show()
        return x

