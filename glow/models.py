import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from utils import weight_norm
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils


class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad : torch.Tensor = None, reverse=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        
        g = - 2 * (inputs - math.log(2) + torch.nn.functional.softplus(- 2 * inputs))
        g = g.view(len(g), -1).sum(-1)
        return torch.tanh(inputs), (g.view(grad.shape) + grad) if grad is not None else g


def f(in_channels, out_channels, hidden_channels, do_actnorm):
    return nn.Sequential(
        modules.Conv2d(in_channels, hidden_channels, do_actnorm=do_actnorm), nn.ReLU(inplace=False),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1], do_actnorm=do_actnorm), nn.ReLU(inplace=False),
        modules.Conv2dZeros(hidden_channels, out_channels))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine", "affineV2"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False,
                 do_actnorm=True):
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.do_actnorm = do_actnorm
        # 1. actnorm
        if do_actnorm:
            self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels, do_actnorm)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels, do_actnorm)
        elif flow_coupling == 'affineV2':
            self.f = f(in_channels // 2, in_channels, hidden_channels, do_actnorm)
            self.register_parameter('scale', nn.Parameter(torch.ones(in_channels // 2, 1, 1)))

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        # assert input.size(1) % 2 == 0
        # 1. actnorm
        if self.do_actnorm:
            z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        else:
            z = input
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        elif self.flow_coupling == 'affineV2':
            """formulation used in RealNVP"""
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = self.scale * torch.tanh(scale)
            z2 = z2 + shift
            z2 = z2 * scale.exp()
            logdet = scale.flatten(1).sum(-1) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        elif self.flow_coupling == 'affineV2':
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = self.scale * torch.tanh(scale)
            z2 = z2 * scale.mul(-1).exp()
            z2 = z2 - shift
            logdet = -scale.flatten(1).sum(-1) + logdet

        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        if self.do_actnorm:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNetV2(nn.Module):
    """FlowNet for non-image data.
    No Squeeze and no Split"""
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False,
                 prior_type='gaussian',
                 vector_mode=False):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        self.prior_type = prior_type 
        H, W, C = image_shape
        self.vector_mode = vector_mode
        # assert H == W == 1, ("Vector-valued data")

        if actnorm_scale == 0:
            do_actnorm = False 
        else:
            do_actnorm = True

        for i in range(L):
            # 1. Squeeze
            if not self.vector_mode:
                C, H, W = C * 4, H // 2, W // 2
                self.layers.append(modules.SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed,
                             do_actnorm=do_actnorm))
                self.output_shapes.append(
                    [-1, C, H, W])
            # 3. Split2d
            if not self.vector_mode:
                if i < L - 1:
                    self.layers.append(modules.Split2d(num_channels=C))
                    self.output_shapes.append([-1, C // 2, H, W])
                    C = C // 2

        if self.prior_type == 'uniform':
            self.layers.append(Tanh())
            self.output_shapes.append(
                [-1, C, H, W])

    def forward(self, input, logdet=0., reverse=False, eps_std=None, l_zs=None):
        if not reverse:
            return self.encode(input, logdet)
        else:
            return self.decode(input, eps_std, l_zs=l_zs)

    def encode(self, z, logdet=0.0):
        l_zs = []
        l_logpz = []
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, modules.Split2d):
                z, logdet, logpz, z2 = layer(z, logdet, reverse=False)
                l_zs.append(z2)
                l_logpz.append(logpz)
            else:
                z, logdet = layer(z, logdet, reverse=False)
        return z, logdet, l_zs, l_logpz

    def decode(self, z, eps_std=None, l_zs=None):
        logdet = 0.
        i_split = 0
        given_z2 = l_zs is not None  # multi-scale z are given
        if given_z2:
            l_zs = l_zs[::-1]
        else:
            l_zs = []
        l_logpz = []

        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                if given_z2:
                    z, logdet, logpz, z2 = layer(z, logdet=logdet, reverse=True, eps_std=eps_std, z2=l_zs[i_split])
                else:
                    z, logdet, logpz, z2 = layer(z, logdet=logdet, reverse=True, eps_std=eps_std)
                    l_zs.append(z2)
                i_split += 1
                l_logpz.append(logpz)
            else:
                z, logdet = layer(z, logdet=logdet, reverse=True)
        return z, logdet, l_zs[::-1], l_logpz



class GlowV2(nn.Module):
    """My modification on GLOW"""
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation='invconv', flow_coupling='affine',
                 LU_decomposed=False, learn_top=False,
                 y_condition=False,
                 y_classes=1, prior_type='gaussian', vector_mode=False,
                 do_logit_transform=False, dequant_offset=False):
        """
        vector_mode: If True, run for image data, else, run for tabular data
                     For tabular data, image_shape should be [1, 1, D]

        for image data, the following options may be needed.
        dequant_offset: likelihood offset due to uniform dequantization
                        likelihood += float(-np.log(256.) * pixels)
        do_logit_transform: preprocess input data using logit transform with bound 0.9
        More information on these preprocessing can be found in
            - https://github.com/chrischute/glow/blob/faffa5ba02f878902a211db76c0bd4ea074b39f7/models/glow/glow.py#L52
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        super().__init__()
        self.flow = FlowNetV2(image_shape=image_shape,
                            hidden_channels=hidden_channels,
                            K=K,
                            L=L,
                            actnorm_scale=actnorm_scale,
                            flow_permutation=flow_permutation,
                            flow_coupling=flow_coupling,
                            LU_decomposed=LU_decomposed,
                            prior_type=prior_type,
                            vector_mode=vector_mode)
        self.y_classes = y_classes
        self.learn_top = learn_top
        self.y_condition = y_condition
        self.prior_type = prior_type
        self.vector_mode = vector_mode
        self.do_logit_transform = do_logit_transform
        self.dequant_offset = dequant_offset
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        # for prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = modules.LinearZeros(
                y_classes, 2 * C)
            self.project_class = modules.LinearZeros(
                C, y_classes)
        # register prior hidden
        # num_device = 1
        # self.register_parameter(
        #     "prior_h",
        #     nn.Parameter(torch.zeros([hparams.Train.batch_size // num_device,
        #                               self.flow.output_shapes[-1][1] * 2,
        #                               self.flow.output_shapes[-1][2],
        #                               self.flow.output_shapes[-1][3]])))

    def prior(self, z):
        if self.prior_type == 'gaussian':
            prior_log_p = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)
            prior_log_p = prior_log_p.view(len(z), -1).sum(-1)
            return prior_log_p
        else:
            raise ValueError(f'Invalid prior type')

    def sample_prior(self, n_sample, device='cpu'):
        shape = [n_sample] + self.flow.output_shapes[-1][1:]
        if self.prior_type == 'gaussian':
            return torch.randn(shape, dtype=torch.float, device=device)
        else:
            raise ValueError(f'{self.prior_type}')

    def sample(self, n_sample, device='cpu'):
        z0 = self.sample_prior(n_sample, device=device)
        d_reverse = self(z=z0, reverse=True, l_zs=None)
        d_reverse['z0'] = z0
        return d_reverse

    def _logit_transform(self, x):
        y = (2 * x - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1 - y).log()
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)
        return y, sldj

    def forward(self, x=None, y_onehot=None, z=None,
                eps_std=None, reverse=False, l_zs=None):
        """
        reverse: if True, z -> x, else x -> z
        """
        if not reverse:
            """ reverse=False : x -> z"""
            assert x is not None
            return self.normal_flow(x, y_onehot)
        else:
            """ reverse=True : z -> x"""
            assert z is not None
            return self.reverse_flow(z, y_onehot, eps_std, l_zs)

    def normal_flow(self, x, y_onehot):
        """From X to Z"""
        logdet = torch.zeros(len(x), device=x.device, dtype=torch.float)
        if self.do_logit_transform:
            x, logdet_ = self._logit_transform(x)
            logdet += logdet_
        # encode
        z, logdet, l_zs, l_logpz = self.flow(x, logdet=logdet, reverse=False)

        # prior
        prior_log_p = self.prior(z)
        lik = logdet + prior_log_p
        for logpz in l_logpz:
            lik += logpz

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        if self.dequant_offset:
            lik -= np.log(256.) * np.prod(x.shape[1:])
        nll = (-lik)
        bpd = nll / np.log(2.) / np.prod(x.shape[1:])

        return {'z': z, 'nll': nll, 'y_logits': y_logits, 'l_zs': l_zs,
                'logdet': logdet, 'prior_log_p': prior_log_p, 'l_logpz': l_logpz, 'bpd': bpd}

    def reverse_flow(self, z, y_onehot, eps_std, l_zs):
        x, logdet, l_zs, l_logpz = self.flow(z, eps_std=eps_std, reverse=True, l_zs=l_zs)
        if self.do_logit_transform:
            # print('WARNING! TODO: logdet of sigmoid transform')
            x = torch.sigmoid(x)  # TODO: logdet of sigmoid transform
        prior_log_p = self.prior(z)
        lik = prior_log_p - logdet
        for logpz_ in l_logpz:
            lik += logpz_
        if self.dequant_offset:
            lik -= np.log(256.) * np.prod(x.shape[1:])

        return {'x': x, 'logdet': logdet, 'l_zs': l_zs, 'l_logpz': l_logpz, 'prior_log_p': prior_log_p,
                'logp': lik}

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    def generate_z(self, img):
        self.eval()
        B = self.hparams.Train.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z,_, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())

    def log_likelihood(self, x):
        # z, nll, y_logits, l_zs = self(x, reverse=False)
        d_out = self(x, reverse=False)
        nll = d_out['nll']
        # nll = d_out['bpd']
        return - nll

    def predict(self, x):
        # z, nll, y_logits, l_zs = self(x, reverse=False)
        d_out = self(x, reverse=False)
        # nll = d_out['nll']
        nll = d_out['bpd']
        return nll

    def train_step(self, x, y=None, optimizer=None, clip_grad=None, scheduler=None):
        optimizer.zero_grad()
        d_out = self(x, reverse=False)
        loss = d_out['nll'].mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
        bpd = d_out['bpd'].mean().item()
        l2_norm = weight_norm(self).item()

        return {'loss': loss.item(), 'glow/train_bpd_': bpd, 'glow/weight_norm_': l2_norm}

    def validation_step(self, x, y=None):
        with torch.no_grad():
            d_out = self(x, reverse=False)
            nll = d_out['nll']
            loss = nll.mean()
            bpd = d_out['bpd'].mean().item()
        d_sample = self.sample(n_sample=8, device=x.device)
        sample = d_sample['x'].detach().cpu()
        sample_img = make_grid(sample, nrow=8, range=(0, 1), normalize=True)
        return {'loss': loss.item(), 'predict': nll.detach().cpu(), 'sample@': sample_img,
                'glow/test_bpd_': bpd}
