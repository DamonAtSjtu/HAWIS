import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
	
# # v1
class ScaleSigner(Function):
	"""take a real value x, output sign(x)*E(|x|)"""
	@staticmethod
	def forward(ctx, input):
		return torch.sign(input) * torch.mean(torch.abs(input))
	@staticmethod
	def backward(ctx, grad_output):
		return grad_output

# v2
# class ScaleSigner(Function):
# 	"""take a real value x, output sign(x)*E(|x|)"""
# 	@staticmethod
# 	def forward(ctx, input):
# 		ctx.save_for_backward(input)
# 		return torch.sign(input) * torch.mean(torch.abs(input))
# 	@staticmethod
# 	def backward(ctx, grad_output):
# 		input, = ctx.saved_tensors
# 		t_clip_up = 1.0
# 		t_clip_down = -1.0
# 		grad_input = grad_output.clone()
# 		grad_input *= (input>t_clip_down).float()
# 		grad_input *= (input<t_clip_up).float()
# 		return grad_input


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

# # v1    
class Quantizer(Function):
	@staticmethod
	def forward(ctx, input):
		scale = 2 ** 1 -1
		return torch.round(input * scale) / scale

	@staticmethod 
	def backward(ctx, grad_output):
		return grad_output, None


# v2   
# class Quantizer(Function):
# 	@staticmethod
# 	def forward(ctx, input):
# 		ctx.save_for_backward(input)
# 		scale = 2 ** 1 -1
# 		return torch.round(input * scale) / scale

# 	@staticmethod 
# 	def backward(ctx, grad_output):
# 		input, = ctx.saved_tensors
# 		t_clip_up = 1.0
# 		t_clip_down = 0.0
# 		grad_input = grad_output.clone()
# 		grad_input *= (input>t_clip_down).float()
# 		grad_input *= (input<t_clip_up).float()
# 		return grad_input, None

def quantize(input):
	return Quantizer.apply(input)

def dorefa_w(w):
	w = scale_sign(w)
	return w

def dorefa_a(input):
	return quantize(torch.clamp(input, 0, 1))

# v-Fracsys
# class QuantSign_FracBNN(torch.autograd.Function):
#     '''
#     Quantize Sign activation to arbitrary bitwidth.
#     Usage: 
#         output = QuantSign_FracBNN.apply(input, bits)
#     '''
#     @staticmethod
#     def forward(ctx, input, bits=4):
#         ctx.save_for_backward(input)
#         input = torch.clamp(input, -1.0, 1.0)
#         delta = 2.0/(2.0**bits-1.0)
#         input = torch.round((input+1.0)/delta)*delta-1.0
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         ''' 
#         Only inputs in the range [-t_clip,t_clip] 
#         have gradient 1. 
#         '''
#         t_clip = 1.0
#         grad_input = grad_output.clone()
#         grad_input *= (input>-t_clip).float()
#         grad_input *= (input<t_clip).float()
#         return grad_input, None

# #v-FracPos
class QuantSign_FracBNN(torch.autograd.Function):
    '''
    Quantize Sign activation to arbitrary bitwidth.
    Usage: 
        output = QuantSign_FracBNN.apply(input, bits)
    '''
    @staticmethod
    def forward(ctx, input, bits=4):
        ctx.save_for_backward(input)
        input = torch.clamp(input, 0, 1.0)
        delta = 1.0/(2.0**bits-1.0)
        input = torch.round(input/delta)*delta
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        ''' 
        Only inputs in the range [-t_clip,t_clip] 
        have gradient 1. 
        '''
        t_clip_up = 1.0
        t_clip_down = 0
        grad_input = grad_output.clone()
        grad_input *= (input>t_clip_down).float()
        grad_input *= (input<t_clip_up).float()
        return grad_input, None


class QuanConv(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(QuanConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

		self.quan_w = dorefa_w
		self.quan_a = dorefa_a

	def forward(self, input):
		w = self.quan_w(self.weight)
		#print('w.mean: ', w.mean())
		x = self.quan_a(input)
		#print('x.abs.mean: ', x.abs().mean())


		output = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
		return output



class QuanConv_first(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super(QuanConv_first, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

		self.quan_w = dorefa_w
		self.quan_a = dorefa_a

	def forward(self, input):
		w = self.quan_w(self.weight)
		#x = self.quan_a(input)
		#print('------------------------------')
		#print("First w.mean: ", w.mean())
		x = input
		#print("First x.abs.mean: ", x.abs().mean())

		output = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
		return output

class QuanLinear_A_multibit(nn.Linear):
	def __init__(self, in_features, out_features, multibit=4, bias=True):
		super(QuanLinear_A_multibit, self).__init__(in_features, out_features, bias)

		self.quan_w = dorefa_w
		self.multibit = multibit
		self.quan_a = QuantSign_FracBNN.apply

	def forward(self, input):
		w = self.quan_w(self.weight)
		x = self.quan_a(input, self.multibit)

		output = F.linear(x, w, bias=None)
		return output



class QuanLinear_W_multibit(nn.Linear):
	def __init__(self, in_features, out_features, multibit=4, bias=True):
		super(QuanLinear_W_multibit, self).__init__(in_features, out_features, bias)

		self.quan_w = QuantSign_FracBNN.apply
		self.multibit = multibit
		self.quan_a = dorefa_a

	def forward(self, input):
		w = self.quan_w(self.weight, self.multibit)
		x = self.quan_a(input)

		output = F.linear(x, w, bias=None)
		return output


class QuanLinear(nn.Linear):
	def __init__(self, in_features, out_features, bias=True):
		super(QuanLinear, self).__init__(in_features, out_features, bias)

		self.quan_w = dorefa_w
		self.quan_a = dorefa_a

	def forward(self, input):
		w = self.quan_w(self.weight)
		#print('Last w.mean: ', w.mean())
		x = self.quan_a(input)
		#print('Last x.abs.mean: ',x.abs().mean())

		output = F.linear(x, w, bias=None)
		return output

# class QuanLinear(nn.Linear):
# 	def __init__(self, in_features, out_features, bias=True):
# 		super(QuanLinear, self).__init__(in_features, out_features, bias)

# 		self.quan_w = dorefa_w
# 		self.quan_a = dorefa_a

# 	def forward(self, input):
# 		w = self.quan_w(self.weight)
# 		x = self.quan_a(input)

# 		output = F.linear(x, w, bias=None)
# 		return output
