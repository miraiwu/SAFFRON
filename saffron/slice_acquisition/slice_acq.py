import sys
from typing import Optional, Sequence
import logging
from torch.autograd import Function
import torch



try:
    import saffron.slice_acq_cuda as slice_acq_cuda
except ImportError:
    try:
        from torch.utils.cpp_extension import load
        import os

        dirname = os.path.dirname(__file__)
        print(dirname)
        slice_acq_cuda = load(
            "slice_acq_cuda",
            [
                os.path.join(dirname, "slice_acq_cuda.cpp"),
                os.path.join(dirname, "slice_acq_cuda_kernel.cu"),
            ],
            verbose=False,
        )
    except:
        logging.warning(
            "Fail to load CUDA extention for slice_acq."
        )


class SliceAcqFunction(Function):
    @staticmethod
    def forward(
        ctx,
        transforms,
        vol,
        vol_mask,
        slices_mask,
        psf,
        slice_shape,
        res_slice,
        need_weight,
        interp_psf,
    ):
        if vol_mask is None:
            vol_mask = torch.empty(0, device=vol.device)
        if slices_mask is None:
            slices_mask = torch.empty(0, device=vol.device)
        outputs = slice_acq_cuda.forward(
            transforms,
            vol,
            vol_mask,
            slices_mask,
            psf,
            slice_shape,
            res_slice,
            need_weight,
            interp_psf,
        )
        ctx.save_for_backward(transforms, vol, vol_mask, slices_mask, psf)
        ctx.interp_psf = interp_psf
        ctx.res_slice = res_slice
        ctx.need_weight = need_weight
        if need_weight:
            return outputs[0], outputs[1]
        else:
            return outputs[0]

    @staticmethod
    def backward(ctx, *args):
        if ctx.need_weight:
            assert len(args) == 2
        grad_slices = args[0]
        transforms, vol, vol_mask, slices_mask, psf = ctx.saved_variables
        interp_psf = ctx.interp_psf
        res_slice = ctx.res_slice
        need_vol_grad = ctx.needs_input_grad[1]
        need_transforms_grad = ctx.needs_input_grad[0]
        outputs = slice_acq_cuda.backward(
            transforms,
            vol,
            vol_mask,
            psf,
            grad_slices.contiguous(),
            slices_mask,
            res_slice,
            interp_psf,
            need_vol_grad,
            need_transforms_grad,
        )
        grad_vol, grad_transforms = outputs
        # print("grad_vol",grad_vol.max())
        # print("grad_transform",grad_transforms.max())
        # sys.exit()
        return grad_transforms, grad_vol, None, None, None, None, None, None, None


class SliceAcqAdjointFunction(Function):
    @staticmethod
    def forward(
        ctx,
        transforms,
        psf,
        slices,
        slices_mask,
        vol_mask,
        vol_shape,
        res_slice,
        interp_psf,
        equalize,
    ):
        if vol_mask is None:
            vol_mask = torch.empty(0, device=slices.device)
        if slices_mask is None:
            slices_mask = torch.empty(0, device=slices.device)
        outputs = slice_acq_cuda.adjoint_forward(
            transforms,
            psf,
            slices,
            slices_mask,
            vol_mask,
            vol_shape,
            res_slice,
            interp_psf,
            equalize,
        )
        vol, vol_weight = outputs
        if equalize:
            ctx.save_for_backward(
                transforms, psf, slices, slices_mask, vol_mask, vol, vol_weight
            )
        else:
            ctx.save_for_backward(transforms, psf, slices, slices_mask, vol_mask)
        ctx.res_slice = res_slice
        ctx.interp_psf = interp_psf
        ctx.equalize = equalize
        return vol

    @staticmethod
    def backward(ctx, grad_vol):
        res_slice = ctx.res_slice
        interp_psf = ctx.interp_psf
        equalize = ctx.equalize
        if equalize:
            (
                transforms,
                psf,
                slices,
                slices_mask,
                vol_mask,
                vol,
                vol_weight,
            ) = ctx.saved_variables
        else:
            transforms, psf, slices, slices_mask, vol_mask = ctx.saved_variables
            vol = vol_weight = torch.empty(0)
        need_slices_grad = ctx.needs_input_grad[2]
        need_transforms_grad = ctx.needs_input_grad[0]
        outputs = slice_acq_cuda.adjoint_backward(
            transforms,
            grad_vol,
            vol_weight,
            vol_mask,
            psf,
            slices,
            slices_mask,
            vol,
            res_slice,
            interp_psf,
            equalize,
            need_slices_grad,
            need_transforms_grad,
        )
        grad_slices, grad_transforms = outputs
        return grad_transforms, None, grad_slices, None, None, None, None, None, None


def slice_acquisition(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    psf: torch.Tensor,
    slice_shape: Sequence,
    res_slice: float,
    need_weight: bool,
    interp_psf: bool,
):
    assert transforms.ndim == 3
    assert vol.ndim == 5
    assert vol_mask is None or vol_mask.ndim == 5
    assert slices_mask is None or slices_mask.ndim == 4

    return SliceAcqFunction.apply(
        transforms,
        vol,
        vol_mask,
        slices_mask,
        psf,
        slice_shape,
        res_slice,
        need_weight,
        interp_psf,
    )


def slice_acquisition_adjoint(
    transforms: torch.Tensor,
    psf: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    vol_shape: Sequence,
    res_slice: float,
    interp_psf: bool,
    equalize: bool,
):
    assert transforms.ndim == 3
    assert slices.ndim == 4
    assert vol_mask is None or vol_mask.ndim == 5
    assert slices_mask is None or slices_mask.ndim == 4

    return SliceAcqAdjointFunction.apply(
        transforms,
        psf,
        slices,
        slices_mask,
        vol_mask,
        vol_shape,
        res_slice,
        interp_psf,
        equalize,
    )
