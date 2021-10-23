import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# v0 torch.sign
# class ScaleSigner(Function):
# 	"""take a real value x, output sign(x)*E(|x|)"""
# 	@staticmethod
# 	def forward(ctx, input):
# 		return torch.sign(input)
# 	@staticmethod
# 	def backward(ctx, grad_output):
# 		return grad_output

# v1 origin
class ScaleSigner(Function):
	"""take a real value x, output sign(x)*E(|x|)"""
	@staticmethod
	def forward(ctx, input):
		return torch.sign(input) * torch.mean(torch.abs(input))
	@staticmethod
	def backward(ctx, grad_output):
		return grad_output

# v2 in-channel scaler;  1X-amplitude backward
# class ScaleSigner(Function):
#     """take a real value x, output sign(x)*E(|x|)"""
#     @staticmethod
#     def forward(ctx, input):
#         if (len(input.shape))==4:
#             scaler = torch.abs(input).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=0,keepdim=True)
#         elif (len(input.shape))==2:
#             scaler = torch.abs(input).mean(dim=0,keepdim=True)
#         else:
#             print("weight tensor wrong")
#         return torch.sign(input) * scaler
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

# v3 in-channel scaler;  same-amplitude backward
# class ScaleSigner(Function):
#     """take a real value x, output sign(x)*E(|x|)"""
#     @staticmethod
#     def forward(ctx, input):
#         if (len(input.shape))==4:
#             scaler = torch.abs(input).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=0,keepdim=True)
#         elif (len(input.shape))==2:
#             scaler = torch.abs(input).mean(dim=0,keepdim=True)
#         else:
#             print("weight tensor wrong")
#         ctx.save_for_backward(scaler)
#         return torch.sign(input) * scaler
#     @staticmethod
#     def backward(ctx, grad_output):
#         scaler, = ctx.saved_tensors
#         return grad_output*scaler

# # v4 in-channel scaler;  same-amplitude normailized by 1X backward
# class ScaleSigner(Function):
#     """take a real value x, output sign(x)*E(|x|)"""
#     @staticmethod
#     def forward(ctx, input):
#         if (len(input.shape))==4:
#             scaler = torch.abs(input).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=0,keepdim=True)
#         elif (len(input.shape))==2:
#             scaler = torch.abs(input).mean(dim=0,keepdim=True)
#         else:
#             print("weight tensor wrong")
#         ctx.save_for_backward(scaler)
#         return torch.sign(input) * scaler
#     @staticmethod
#     def backward(ctx, grad_output):
#         scaler, = ctx.saved_tensors
#         scaler /= torch.mean(scaler)
#         return grad_output*scaler

# # v2 out-channel scaler;  1X-amplitude backward
# class ScaleSigner(Function):
#     """take a real value x, output sign(x)*E(|x|)"""
#     @staticmethod
#     def forward(ctx, input):
#         if (len(input.shape))==4:
#             scaler = torch.abs(input).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=1,keepdim=True)
#         elif (len(input.shape))==2:
#             scaler = torch.abs(input).mean(dim=1,keepdim=True)
#         else:
#             print("weight tensor wrong")
#         return torch.sign(input) * scaler
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

# v4 out-channel scaler;  same-amplitude normailized by 1X backward
# class ScaleSigner(Function):
#     """take a real value x, output sign(x)*E(|x|)"""
#     @staticmethod
#     def forward(ctx, input):
#         if (len(input.shape))==4:
#             scaler = torch.abs(input).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=1,keepdim=True)
#         elif (len(input.shape))==2:
#             scaler = torch.abs(input).mean(dim=1,keepdim=True)
#         else:
#             print("weight tensor wrong")
#         ctx.save_for_backward(scaler)
#         return torch.sign(input) * scaler
#     @staticmethod
#     def backward(ctx, grad_output):
#         scaler, = ctx.saved_tensors
#         scaler /= torch.mean(scaler)
#         return grad_output*scaler

def scale_sign(input):
	return ScaleSigner.apply(input)

class Quantizer(Function):
	@staticmethod
	def forward(ctx, input):
		scale = 2 ** 1 -1
		return torch.round(input * scale) / scale

	@staticmethod 
	def backward(ctx, grad_output):
		return grad_output, None

def quantize(input):
	return Quantizer.apply(input)

def dorefa_w(w):
	# v1 origin
	w = scale_sign(w) 
	# # v0-5 in-channel scaler, auto grad
	# if (len(w.shape))==4:
	# 	scaler = torch.abs(w).mean(dim=3,keepdim=True).mean(dim=2,keepdim=True).mean(dim=0,keepdim=True)
	# elif (len(w.shape))==2:
	# 	scaler = torch.abs(w).mean(dim=0,keepdim=True)
	# else:
	# 	print("weight tensor wrong")
	# w = scale_sign(w)*scaler
	return w

def dorefa_a(input):
    # origin
	return quantize(torch.clamp(input, 0, 1))
	# QA-v2
	#return quantize(torch.clamp(input+0.5, 0, 1))
	# QA-v3
	#return quantize(torch.clamp(input+0.05, 0, 1))


class _quantize_dac_weight(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, delta_x):
        # ctx is a context object that can be used to stash information for backward computation
        ctx.delta_x = delta_x
        output = torch.round(input/ctx.delta_x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.delta_x
        return grad_input, None

