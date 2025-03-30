import torch
import torch.nn as nn
import torch.quantization
from einops import rearrange
import math
import time

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quant=True, calib=False, qact=False, bit_width=4):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.quant = quant
        self.bit_width = bit_width
        self.calib = calib
        self.qact = qact
        self.act_scale = None
        self.act_zeropoint = None
        self.outputs = []
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weights(self, weights, num_bits):
        min_val = weights.min()
        max_val = weights.max()

        qmin = -(2**(num_bits - 1))
        qmax = 2**(num_bits - 1) - 1

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        quantized = zero_point + weights / scale
        quantized.clamp_(qmin, qmax).round_()

        return quantized, scale, zero_point

    def dequantize_tensor(self, quantized_tensor, scale, zero_point):
        dequantized = scale * (quantized_tensor - zero_point)
        return dequantized

    def determine_range(self, outputs):
        min_val = min(output.min() for output in outputs)
        max_val = max(output.max() for output in outputs)
        return min_val, max_val

    def calculate_quantization_params(self, min_val, max_val, num_bits):
        qmin = -(2**(num_bits - 1))
        qmax = 2**(num_bits - 1) - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        return scale, zero_point

    def quantize_tensor(self, tensor, scale, zero_point, num_bits):
        qmin = -(2**(num_bits - 1))
        qmax = 2**(num_bits - 1) - 1

        quantized = zero_point + tensor / scale
        quantized.clamp_(qmin, qmax).round_()

        return quantized

    def forward(self, input):
        if(self.calib==True):
            quantized_weights, scale, zero_point = self.quantize_weights(self.weight, self.bit_width)
            out = nn.functional.linear(input, quantized_weights, self.bias)
            out = self.dequantize_tensor(out, scale, zero_point)
            self.outputs.append(out)
            min_val, max_val = self.determine_range(self.outputs)
            self.act_scale, self.act_zeropoint = self.calculate_quantization_params(min_val, max_val, 4)
            return out
        if(self.quant==True):
            if(self.qact==False):
                quantized_weights, scale, zero_point = self.quantize_weights(self.weight, self.bit_width)
                out = nn.functional.linear(input, quantized_weights, self.bias)
                out = self.dequantize_tensor(out, scale, zero_point)
                return out
            elif(self.qact==True):
                quantized_weights, scale, zero_point = self.quantize_weights(self.weight, self.bit_width)
                out = nn.functional.linear(input, quantized_weights, self.bias)
                out = self.dequantize_tensor(out, scale, zero_point)
                out = self.quantize_tensor(out, self.act_scale, self.act_zeropoint, 4)
                out = self.dequantize_tensor(out, self.act_scale, self.act_zeropoint)
                return out
        if(self.quant==False):
            return nn.functional.linear(input, self.weight, self.bias)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            CustomLinear(dim, hidden_dim, quant=True, qact=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CustomLinear(hidden_dim, dim, quant=True, qact=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = CustomLinear(dim, inner_dim * 3, bias = False, quant=True)

        self.to_out = nn.Sequential(
            CustomLinear(inner_dim, dim, quant=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = CustomLinear(channels[-1], num_classes, bias=False, quant=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def MobileViT_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def MobileViT_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def MobileViT_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_calib_true_for_all_custom_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, CustomLinear):
            module.calib = True

def set_calib_false_for_all_custom_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, CustomLinear):
            module.calib = False
'''
if __name__ == '__main__':

    img = torch.randn(5, 3, 256, 256)

    vit = MobileViT_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = MobileViT_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = MobileViT_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
'''
~             
