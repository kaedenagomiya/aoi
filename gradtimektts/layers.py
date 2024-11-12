import torch
from torch import nn

from .base import BaseModule
#from einops import rearrange

class Mish(BaseModule):

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class TimeFreqKernelConv2d(BaseModule):
    """
    Conv2d API
    -----------
    torch.nn.Conv2d(in_channels, out_channels,
                    kernel_size,stride=1, padding=0, 
                    dilation=1, groups=1, bias=True,
                    padding_mode='zeros', device=None, dtype=None)
    Ref.[nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 tf_kernel=(3,3),
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        # def __init__(self, dim, mixer_kernel, dilation = 1):
        super(TimeFreqKernelConv2d, self).__init__()

        """
        self.conv1 = nn.Conv2d(in_channels,in_channels,
                               kernel_size,stride,padding,dilation,
                               groups=in_channels,bias=bias)
        """
        f, t = tf_kernel
        """
        self.tfkconv_f = nn.Conv2d(in_channels,in_channels,
                                    kernel_size=(f, 1), 
                                    padding='same',
                                    groups=in_channels,
                                    dilation=dilation)
        """
        self.tfkconv_t = nn.Conv2d(in_channels,in_channels,
                                    kernel_size=(1, t),
                                    padding='same',
                                    groups=in_channels, 
                                    dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1,
                                    bias=bias)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.tfkconv_f(x) + self.tfkconv_t(x)
        x = self.tfkconv_t(x)
        x = self.pointwise(x)
        return x


class SeparableBlock(BaseModule):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            # (in_channel, out_channel, tf_kernel=(f,t))
            TimeFreqKernelConv2d(dim, dim_out, tf_kernel=(3,3), padding=1),
            nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class SeparableResnetBlock(BaseModule):

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, dim_out)

        self.block1 = SeparableBlock(dim, dim_out, groups=groups)
        self.block2 = SeparableBlock(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_q = torch.nn.Conv2d(dim, self.hidden_dim, 1, bias=False)
        self.to_k = torch.nn.Conv2d(dim, self.hidden_dim, 1, bias=False)
        self.to_v = torch.nn.Conv2d(dim, self.hidden_dim, 1, bias=False)
        self.to_out = torch.nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.to_q(x).reshape((b, self.heads, -1, h * w))  # (b,heads,d,h*w)
        k = self.to_k(x).reshape((b, self.heads, -1, h * w))  # (b,heads,d,h*w)
        v = self.to_v(x).reshape((b, self.heads, -1, h * w))  # (b,heads,e,h*w)
        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.permute(0, 1, 3, 2))  # (b,heads,d,e)
        out = torch.matmul(context.permute(0, 1, 3, 2), q)  # (b,heads,e,n)
        out = out.reshape(b, self.hidden_dim, h, w)
        return self.to_out(out)
