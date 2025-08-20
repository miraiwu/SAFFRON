# -*- coding: utf-8 -*-
# ------------------------------------
# @Author  : Jiangjie Wu
# @FileName: models.py
# @Software: PyCharm
# @Email   : wujj@shanghaitech.edu.cn
# @Date    : 2024-03-30

from typing import Callable, Dict, Optional, Tuple, Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..svr.reconstruction import cg
from ..transform import (
    RigidTransform,
    mat_update_resolution,
    ax_update_resolution,
    mat2axisangle,
    point2mat,
    mat2point,
)

from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint
from saffron.tool import gaussian_blur,ncc_loss,Volume, Stack,get_PSF

import torchvision.models as tvm
from saffron.svr.SVRmamba import MambaEncoder
from saffron.srr.model import ResidualUNet3D


# main models

USE_MASK = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, d_in: int) -> None:
        super().__init__()
        num_w = d_model // 2 // d_in
        self.num_pad = d_model - num_w * 2 * d_in
        w = 1e-3 ** torch.linspace(0, 1, num_w)
        self.w = nn.Parameter(w.view(1, -1, 1).repeat(1, 1, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.cat((torch.sin(self.w * x), torch.cos(self.w * x)), 1)
        x = x.flatten(1)
        if self.num_pad:
            x = F.pad(x, (0, self.num_pad))
        return x

class ResNet(nn.Module):
    def __init__(
        self, n_res: int, d_model: int, d_in: int = 1, pretrained: bool = False
    ) -> None:
        super().__init__()
        resnet_fn = getattr(tvm, "resnet%d" % n_res)
        model = resnet_fn(
            # pretrained=pretrained,
            norm_layer=lambda x: nn.BatchNorm2d(x, track_running_stats=False),
        )
        model.fc = nn.Linear(model.fc.in_features, d_model)
        if not pretrained:
            model.conv1 = nn.Conv2d(
                d_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.model = model
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = x.expand(-1, 3, -1, -1)
        return self.model(x)



class SAFFRON(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True,test=False):
        super().__init__()
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        # self.attn = None
        self.test = test
        self.iqa_score = None
        self.n_iter = n_iter
        # self.ncc_threshold = [0.6,0.7,0.8]
        self.ncc_threshold = 0.6
        self.device1 =  torch.device("cuda:0")
        self.device2 =  torch.device("cuda:1")

        self.svrnet1 = SVRMamba(
            n_layers=4,
            # n_layers=1,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRMamba(
            n_layers=4 * 2,
            # n_layers=2 ,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )
        self.srrnet = SRRUNET(in_channels=1, out_channels=1, final_sigmoid=False, f_maps=16, layer_order='gcr',
                 # num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1,
                 num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1,)

        if iqa:
            self.srr = SRR(n_iter=2)


    def forward(self, data):
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]

        thetas = []
        volumes = []
        trans = []
        iqa_scores = []
        ncc_idxs = []
        loss_slices = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0  # + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None
        mask_stacks = gaussian_blur(stacks, 1.0, 3) > 0 if USE_MASK else None

        for i in range(self.n_iter):
            with torch.no_grad():
                svrnet = self.svrnet2 if i else self.svrnet1
                theta, iqa_score = svrnet(
                    theta,
                    stacks,
                    positions,
                    None if ((volume is None) or (not self.vol)) else volume.detach(),
                    params,
                )

                _trans = RigidTransform(point2mat(theta))

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = slice_acquisition_adjoint(
                    mat,
                    params["psf"],
                    stacks,
                    mask_stacks,
                    None,
                    params["volume_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    equalize=True,
                )



            if self.iqa:
                # ncc = []
                # volume = volume.to(self.device2)
                # iqa_score = iqa_score.to(self.device2)

                # stacks=stacks[ncc_idx]
                # mask_stacks=stacks[ncc_idx]
                # _trans = _trans[ncc_idx]
                # mat = mat[ncc_idx]
                # theta = theta[ncc_idx]
                # iqa_scores=iqa_scores[ncc_idx]
                if self.test:

                    with torch.set_grad_enabled(self.test):
                        mat = nn.Parameter(mat,requires_grad=True)
                        optimizer = torch.optim.SGD([mat], lr=0.01, momentum=0.9)

                        print(mat[10])
                        volume = self.srr(
                            mat,
                            stacks,
                            volume,
                            params,
                            iqa_score.view(-1, 1, 1, 1),
                            mask_stacks,
                            None,
                            optimizer,
                        )
                        print(mat.requires_grad)
                        print(mat[10])
                else:
                    volume = self.srr(
                        mat,
                        stacks,
                        volume,
                        params,
                        iqa_score.view(-1, 1, 1, 1),
                        mask_stacks,
                        None,
                    )
                    volume = self.srrnet(volume)

                self.iqa_score = iqa_score.detach()
                iqa_scores.append(iqa_score)
                # ncc_idxs.append(ncc_idx)
                # loss_slices.append(r)
            thetas.append(theta)
            trans.append(_trans)
            volumes.append(volume)

        return trans, volumes, thetas,iqa_scores




class SRRUNET(nn.Module):
    def __init__(self,
        in_channels, out_channels, final_sigmoid=False, f_maps=16, layer_order='gcr',
        num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1,
        conv_upscale=2, upsample='default', dropout_prob=0.1,):
        super().__init__()
        self.unet = ResidualUNet3D(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             conv_upscale=conv_upscale,
                                             upsample=upsample,
                                             dropout_prob=dropout_prob)

    def forward(self, volume):
        dvolume = self.unet(volume)

        return volume+dvolume




class SVRMamba(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        n_channels=2,
    ):
        super().__init__()
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)

        self.encoder = MambaEncoder(
            n_layers=n_layers,
            # n_layers=4,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        # self.encoder = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=d_model, # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,    # Local convolution width
        #     expand=2,    # Block expansion factor
        # )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask=None):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    False,
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        if isinstance(self.img_encoder, ResNet):
            slices = self.pos_augment(slices, y)
        else:  # ViT
            if y is not None:
                slices = torch.cat([slices, y], 1)
        x = self.img_encoder(slices)
        # x, attn = self.encoder(x, pe, attn_mask)
        x = self.encoder(x, pe)
        dtheta = self.fc(x)

        score = self.fc_score(x)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score


class SRR(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        srr_net=None,
        tol: float = 0.0,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.srr_net = srr_net
        self.tol = tol
        self.output_relu = output_relu

    def forward(
        self,
        transforms: torch.Tensor,
        slices: torch.Tensor,
        volume: torch.Tensor,
        params: Dict,
        p: Optional[torch.Tensor] = None,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
        optimizer=None,
    ) -> torch.Tensor:
    # ) -> [torch.Tensor,torch.Tensor]:
        At = lambda x: self.At(transforms, x, params, slices_mask, vol_mask)
        AtA = lambda x: self.AtA(transforms, x, p, params, slices_mask, vol_mask)


        b = At(slices * p if p is not None else slices)
        volume = cg(AtA, b, volume, self.n_iter, self.tol,optimizer)

        if self.output_relu:
            # volume = F.relu(volume, True)
            # volume = self.srr_net(volume)
            # r = slices - self.A(transforms, volume, params, slices_mask, vol_mask) * p
            # return  volume,r
            return F.relu(volume, True)

        else:
            return volume
            # volume = self.srr_net(volume)
            # r = slices - self.A(transforms, volume, params, slices_mask, vol_mask) * p
            # return volume, r

    def A(
            self,
            transforms: torch.Tensor,
            x: torch.Tensor,
            params: Dict,
            slices_mask: Optional[torch.Tensor] = None,
            vol_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return slice_acquisition(
                transforms,
                x,
                vol_mask,
                slices_mask,
                params["psf"],
                params["slice_shape"],
                params["res_s"] / params["res_r"],
                False,
                False,
            )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        p: Optional[torch.Tensor],
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        slices = self.A(transforms, x, params, slices_mask, vol_mask)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, params, slices_mask, vol_mask)
        return vol

class SCC(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        mu: float = 0.0,
        average_init: bool = False,
        output_relu: bool = True,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.mu = mu
        self.output_relu = output_relu
        self.use_mask = use_mask
        self.average_init = average_init

    def forward(
        self,
        stack: Stack,
        volume: Volume,
        p: Union[Stack, torch.Tensor, None] = None,
        z: Union[Stack, torch.Tensor, None] = None,
        psf: Optional[torch.Tensor] = None,
    ) -> Volume:
        transforms, res_s, res_r, s_thick, psf = _parse_stack_volume(stack, volume, psf)

        params = {
            "psf": psf,
            "slice_shape": stack.shape[-2:],
            "res_s": res_s,
            "res_r": res_r,
            "volume_shape": volume.shape[-3:],
        }

        slices_mask = stack.mask if self.use_mask else None
        vol_mask = volume.mask[None, None] if self.use_mask else None

        if isinstance(p, Stack):
            p = p.slices
        if isinstance(z, Stack):
            z = z.slices

        # A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(
            transforms, x, vol_mask, slices_mask, p, params, self.mu, z
        )
        y = stack.slices
        if self.average_init:
            x = slice_acquisition_adjoint(
                transforms,
                psf,
                y,
                slices_mask,
                vol_mask,
                volume.shape[-3:],
                res_s / res_r,
                False,
                equalize=True,
            )
        else:
            x = volume.image[None, None]

        b = At(y * p if p is not None else y)
        if self.mu and z is not None:
            b = b + self.mu * z
        x = cg(AtA, b, x, self.n_iter, self.tol)

        if self.output_relu:
            x = F.relu(x, True)

        return cast(Volume, Volume.like(volume, image=x[0, 0], deep=False))

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        slices_mask: Optional[torch.Tensor],
        vol_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
        params: Dict,
        mu: float,
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol




def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.flatten(), y.flatten())

def cg(
    A: Callable, b: torch.Tensor, x0: torch.Tensor, n_iter: int, tol: float = 0.0,optimizer=None
) -> torch.Tensor:
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    loss = 0.5*(r**2).mean()
    # loss = 0.5*dot(r, r)
    while True:
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss", loss)
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1

        #------
        if optimizer is not None:
            error = b.detach() - A(x.detach())
            # loss = 0.5 * (error * error).mean()
            loss = (error**2/2).mean()
            # loss = 0.5*dot(error, error)
            # (v_out - v) ** 2 / 2 ).mean()


        #----
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        if dot_r_r_new <= tol:
            return x
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new


def simulate_slices(
    slices: Stack,
    volume: Volume,
    return_weight: bool = False,
    use_mask: bool = False,
    psf: Optional[torch.Tensor] = None,
) -> Union[Tuple[Stack, Stack], Stack]:
    slices_transform_mat, res_s, res_r, s_thick, psf = _parse_stack_volume(
        slices, volume, psf
    )

    outputs = slice_acquisition(
        slices_transform_mat,
        volume.image[None, None],
        volume.mask[None, None] if use_mask else None,
        slices.mask if use_mask else None,
        psf,
        slices.shape[-2:],
        res_s / res_r,
        return_weight,
        False,
    )

    if return_weight:
        slices_sim, weight = outputs
        return (
            Stack.like(slices, slices=slices_sim, deep=False),
            Stack.like(slices, slices=weight, deep=False),
        )
    else:
        return Stack.like(slices, slices=outputs, deep=False)

def _parse_stack_volume(
    stack: Stack, volume: Volume, psf: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float, float, float, torch.Tensor]:
    eps = 1e-3
    assert (
        abs(volume.resolution_x - volume.resolution_y) < eps
        and abs(volume.resolution_x - volume.resolution_z) < eps
    ), "input volume should be isotropic!"
    assert (
        abs(stack.resolution_x - stack.resolution_y) < eps
    ), "input slices should be isotropic!"

    res_s = float(stack.resolution_x)
    res_r = float(volume.resolution_x)
    s_thick = float(stack.thickness)

    slices_transform = stack.transformation
    volume_transform = volume.transformation
    slices_transform = volume_transform.inv().compose(slices_transform)

    slices_transform_mat = mat_update_resolution(slices_transform.matrix(), 1, res_r)

    if psf is None:
        psf = get_PSF(
            # r_max=5,
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
            device=volume.device,
        )

    return slices_transform_mat, res_s, res_r, s_thick, psf