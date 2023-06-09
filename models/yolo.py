import argparse
from models.experimental import *
class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  
        super(Detect, self).__init__()
        self.stride = None  
        self.nc = nc  
        self.no = nc + 5 
        self.nl = len(anchors)  
        self.na = len(anchors[0]) // 2  
        self.grid = [torch.zeros(1)] * self.nl  
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  
        self.export = False  
    def forward(self, x):
        z = []  
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
class Model(nn.Module):
    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):  
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg  
        else:  
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  
        if nc:
            self.md['nc'] = nc 
        self.model, self.save = parse_model(self.md, ch=[ch]) 
        m = self.model[-1]  
        m.stride = torch.tensor([64 / x.shape[-2] for x in self.forward(torch.zeros(1, ch, 64, 64))])  
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        torch_utils.initialize_weights(self)
        self._initialize_biases()  
        torch_utils.model_info(self)
        print('')
    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  
            s = [0.83, 0.67] 
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]), 
                                    torch_utils.scale_img(x, s[1]),  
                                    )):
                y.append(self.forward_once(xi)[0])
            y[1][..., :4] /= s[0]  
            y[1][..., 0] = img_size[1] - y[1][..., 0]  
            y[2][..., :4] /= s[1]  
            return torch.cat(y, 1), None 
        else:
            return self.forward_once(x, profile)
    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1: 
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % ( m.np, dt[-1], m.type))
            x = m(x)  
            y.append(x if m.i in self.save else None)  
        if profile:
            print('%.1fms total' % sum(dt))
        return x
    def _initialize_biases(self, cf=None): 
        m = self.model[-1]  
        for f, s in zip(m.f, m.stride):  
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  
            b[:, 4] += math.log(8 / (640 / s) ** 2)  
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _print_biases(self):
        m = self.model[-1]  
        for f in sorted([x % m.i for x in m.f]):  
            b = self.model[f].bias.detach().view(m.na, -1).T 
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))
    def fuse(self): 
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  
                m.bn = None 
                m.forward = m.fuseforward  
        torch_utils.model_info(self)
def parse_model(md, ch):  
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    na = (len(anchors[0]) // 2)  
    no = na * (nc + 5) 
    layers, save, c2 = [], [], ch[-1]  
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']): 
        m = eval(m) if isinstance(m, str) else m  
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a 
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  
        t = str(m)[8:-2].replace('__main__.', '')  
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  
        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args)) 
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  
    device = torch_utils.select_device(opt.device)
    model = Model(opt.cfg).to(device)
    model.train()