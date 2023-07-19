import os
from collections import namedtuple, OrderedDict
import numpy as np

import pyro
import pyro.optim as optim
# from pyro.infer import SVI, TraceGraph_ELBO
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.contrib.examples.multi_mnist as multi_mnist
import torch
import torch.nn as nn
from torch.nn.functional import relu, tanh, sigmoid, softplus, grid_sample, affine_grid, dropout

from sympy.solvers import solve
from sympy import Symbol, var
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import diop_solve
from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic


torch.set_default_dtype(torch.float32)

def solve_padding_dilation(h_in, h_out, kernel_size=3, stride=1):
    H_in = Symbol("H_in", integer=True, positive=True)
    H_out = Symbol("H_out", integer=True, positive=True)
    padding = var("padding", integer=True, positive=True)
    dilation = var("dilation", integer=True, positive=True)


    equation = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride  + 1 - H_out
    # equation.subs({H_in:32, H_out:40, padding:0})
    
    for i in range(100):
        solution = diophantine(equation.subs({H_in:h_in, H_out:h_out, padding:i}))
                
        if len(solution) == 0:
            continue

        dilation_val = list(solution)[0][0]
        
        
        if dilation_val > 0:
            padding_val = i

            print("solution: ", solution, "dilation_val: ", dilation_val)
            break

    return padding_val, dilation_val



# Create the neural network. This takes a latent code, z_what, to pixel intensities.
class Decoder(nn.Module):
    def __init__(self, use_cuda=True, device=None, 
        z_what_latent_size=100, obj_height=100, obj_width=100, 
        *args, **kargs):
        super().__init__(*args, **kargs)
        self.obj_width = obj_width
        self.obj_height = obj_height
        
        # self.l1 = nn.Linear(z_what_latent_size, 1024*1*1)
        # self.l2 = nn.Linear(1024, 512)
        # self.l3 = nn.Linear(512, 1024)
        # self.u1 = nn.Unflatten(1, (1024, 1, 1))
        
        # # 1x1 -> 2x2
        # self.t1  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.c1a = nn.Conv2d(512, 256, 3, padding="same")
        
        
        # # 2x2 -> 4x4
        # self.t2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.c2a = nn.Conv2d(128, 64, 3, padding="same")
        
        # # 4x4 -> 8x8
        # self.t3  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.c3a = nn.Conv2d(32, 16, 3, padding="same")
        
        # # 8x8 -> 16x16
        # self.t4  = nn.ConvTranspose2d(16, 8, 2, stride=2)
        # self.c4a = nn.Conv2d(8, 4, 3, padding="same")

        
        # # 16x16 -> 32x32
        # self.t5  = nn.ConvTranspose2d(4, 2, 2, stride=2)
        # self.c5a = nn.Conv2d(2, 1, 3, padding="same")

        # # # 32x32 -> 64x64
        # # self.t6  = nn.ConvTranspose2d(1, 2, 2, stride=2)
        # # self.c6a = nn.Conv2d(2, 1, 3, padding="same")

        # # # 64x64 -> 128x128
        # # self.t7  = nn.ConvTranspose2d(1, 2, 2, stride=2)
        # # self.c7a = nn.Conv2d(2, 1, 3, padding="same")

        # # padding, dilation = solve_padding_dilation(128, obj_width, kernel_size=3, stride=1)
        # # self.cn = nn.Conv2d(1, 1, 3, stride=1, padding=padding, dilation=dilation)


        # self.fn = nn.Flatten()
        # self.ln = nn.Linear(32*32, obj_width*obj_height)
        # # self.un = nn.Unflatten(1, (1, obj_height, obj_width))

        self.l1 = nn.Linear(z_what_latent_size, 2000)
        self.l2 = nn.Linear(2000, 4000)
        self.l3 = nn.Linear(4000, 2000)
        self.l4 = nn.Linear(2000, obj_height*obj_width*2)

        
        if use_cuda:
            self.cuda(device=device)

    def forward(self, z_what):
        obj_width = self.obj_width
        obj_height = self.obj_height

        # # h = relu(self.l1(z_what))
        # # h = self.l1(z_what)
        # # h = relu(h)
        # # h = self.l2(h)
        # # h = sigmoid(h)
        
        # h = self.l1(z_what)
        # h = tanh(h)
        # h = self.l2(h)
        # h = tanh(h)
        # h = self.l3(h)
        # h = tanh(h)
        # h = self.u1(h)
        
        # h = self.t1(h)
        # h = tanh(h)
        # h = self.c1a(h)
        # h = tanh(h)

        
        # h = self.t2(h)
        # h = tanh(h)
        # h = self.c2a(h)
        # h = tanh(h)
        
        # h = self.t3(h)
        # h = tanh(h)
        # h = self.c3a(h)
        # h = tanh(h)
                 
        # h = self.t4(h)
        # h = tanh(h)
        # h = self.c4a(h)
        # h = tanh(h)
                 
        # h = self.t5(h)
        # h = tanh(h)
        # h = self.c5a(h)
        # h = tanh(h)

        # # h = self.t6(h)
        # # h = tanh(h)
        # # h = self.c6a(h)
        # # h = tanh(h)

        # # h = self.t7(h)
        # # h = tanh(h)
        # # h = self.c7a(h)
        # # h = tanh(h)

        
        # # h = self.cn(h)
        # h = self.fn(h)
        # h = self.ln(h)
        # # h = self.un(h)

        h = self.l1(z_what)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        h = self.l2(h)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        h = self.l3(h)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        h = self.l4(h)
        # h = tanh(h)
        
        
        obj_loc = sigmoid(h[..., :obj_width*obj_height]/1000)
        # obj_loc = h[..., :obj_width*obj_height]
        obj_scale = sigmoid(h[..., obj_width*obj_height:]/1000)*0.299 + 0.001
        
        # return sigmoid(self.l2(h))
        return obj_loc, obj_scale

# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
class Encoder(nn.Module):
    def __init__(self, use_cuda=True, device=None, 
        z_what_latent_size=100, obj_height=100, obj_width=100, 
        *args, **kargs):
        super().__init__(*args, **kargs)

        self.z_what_latent_size = z_what_latent_size
        
        # # self.l1 = nn.Linear(obj_height*obj_width, 1024)
        # # self.l2 = nn.Linear(1024, obj_height*obj_width)
        
        # self.l1 = nn.Linear(obj_height*obj_width, 128*128)
        # self.u1 = nn.Unflatten(1, (1, 128, 128))
        
        # # padding, dilation = solve_padding_dilation(obj_width, 128, kernel_size=3, stride=1)
        # # self.c0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        
        # # 128x128 -> 64x64
        # self.c1 = nn.Conv2d(1, 2, 3, 1, padding="same")
        # self.p1 = nn.MaxPool2d(2, stride=2)
        # self.c1a = nn.Conv2d(2, 4, 3, 1, padding="same")
        
        
        # # 64x64 -> 32x32
        # self.c2 = nn.Conv2d(4, 8, 3, 1, padding="same")
        # self.p2 = nn.MaxPool2d(2, stride=2)
        # self.c2a = nn.Conv2d(8, 16, 3, 1, padding="same")
        
        
        # # 32x32 -> 16x16
        # self.c3 = nn.Conv2d(16, 32, 3, 1, padding="same")
        # self.p3 = nn.MaxPool2d(2, stride=2)
        # self.c3a = nn.Conv2d(32, 64, 3, 1, padding="same")

        
        # # 16x16 -> 8x8
        # self.c4 = nn.Conv2d(64, 128, 3, 1, padding="same")
        # self.p4 = nn.MaxPool2d(2, stride=2)
        # self.c4a = nn.Conv2d(128, 256, 3, 1, padding="same")

        
        
        # # 8x8 -> 4x4
        # self.c5 = nn.Conv2d(256, 512, 3, 1, padding="same")
        # self.p5 = nn.MaxPool2d(2, stride=2)
        # self.c5a = nn.Conv2d(512, 1024, 3, 1, padding="same")
        
        
        # # 4x4 -> 2x2
        # self.c6 = nn.Conv2d(1024, 512, 3, 1, padding="same")
        # self.p6 = nn.MaxPool2d(2, stride=2)
        # self.c6a = nn.Conv2d(512, 1024, 3, 1, padding="same")
        
        # # 2x2 -> 1x1
        # self.c7 = nn.Conv2d(1024, 512, 3, 1, padding="same")
        # self.p7 = nn.MaxPool2d(2, stride=2)
        # self.c7a = nn.Conv2d(512, 1024, 3, 1, padding="same")
        
        # self.fn = nn.Flatten()
        # self.ln = nn.Linear(1024*1*1, 512)
        # self.ln1 = nn.Linear(512, 1024)
        # self.ln2 = nn.Linear(1024, z_what_latent_size*2)


        self.l1 = nn.Linear(obj_height*obj_width, 2000)
        self.l2 = nn.Linear(2000, 4000)
        self.l3 = nn.Linear(4000, 2000)
        self.l5 = nn.Linear(2000, z_what_latent_size*2)
        
        
        if use_cuda:
            self.cuda(device=device)

    def forward(self, data):
        z_what_latent_size = self.z_what_latent_size


        # # h = relu(self.l1(data))
        # # h = self.l1(data)
        # # h = relu(h)
        # # h = self.l2(h)
        
        # h = self.l1(data)
        # h = tanh(h)
        # h = self.u1(h)
        # # h = self.c0(h)
        # h = tanh(h)
        
        # h = self.c1(h)
        # h = tanh(h)
        # h = self.p1(h)
        # h = self.c1a(h)
        # h = tanh(h)

        
        # h = self.c2(h)
        # h = tanh(h)
        # h = self.p2(h)
        # h = self.c2a(h)
        # h = tanh(h)
        
        # h = self.c3(h)
        # h = tanh(h)
        # h = self.p3(h)
        # h = self.c3a(h)
        # h = tanh(h)
        
        # h = self.c4(h)
        # h = tanh(h)
        # h = self.p4(h)
        # h = self.c4a(h)
        # h = tanh(h)
        
        # h = self.c5(h)
        # h = tanh(h)
        # h = self.p5(h)
        # h = self.c5a(h)
        # h = tanh(h)
        
        # h = self.c6(h)
        # h = tanh(h)
        # h = self.p6(h)
        # h = self.c6a(h)
        # h = tanh(h)
        
        # h = self.c7(h)
        # h = tanh(h)
        # h = self.p7(h)
        # h = self.c7a(h)
        # h = tanh(h)

        
        # h = self.fn(h)
        # h = self.ln(h)
        # h = self.ln1(h)
        # a = self.ln2(h)

        h = self.l1(data)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        h = self.l2(h)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        h = self.l3(h)
        h = relu(h/1000)
        h = dropout(h, p=0.5)
        # h = self.l4(h)
        # h = tanh(h)
        a = self.l5(h)
        # a = tanh(h)

        
        # z_loc = sigmoid(a[:, 0:z_what_latent_size])
        # z_loc = sigmoid(a[:, 0:z_what_latent_size])
        z_loc = relu(a[:, 0:z_what_latent_size]/1000)
        # z_loc = a[:, 0:z_what_latent_size]
        z_scale = sigmoid(a[:, z_what_latent_size:]/1000) * 0.299 + 0.001
        return z_loc, z_scale
        
      
        
## 一个 compartment 一个细胞就不需要 LSTM
class Compartment(nn.Module):
    def __init__(self, use_cuda=True, device=None, 
        height=256, width=256, 
        n_grid_height=8, n_grid_width=8, maximum_in_grid=3, 
        *args, **kargs):
        super().__init__(*args, **kargs)

        self.n_grid_height = n_grid_height
        self.n_grid_width = n_grid_width
        self.maximum_in_grid = maximum_in_grid
        self.device = device
        self.hidden_size = 256
        self.num_layers = 1
        
        # 256x256 -> 128x128
        self.c1 = nn.Conv2d(1, 8, 3, 1, padding="same")
        self.p1 = nn.MaxPool2d(2, stride=2)
        
        # 128x128 -> 64x64
        self.c2 = nn.Conv2d(8, 32, 3, 1, padding="same")
        self.p2 = nn.MaxPool2d(2, stride=2)
        
        # 64x64 -> 32x32
        self.c3 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.p3 = nn.MaxPool2d(2, stride=2)
        
        # 32x32 -> 16x16
        self.c4 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.p4 = nn.MaxPool2d(2, stride=2)
        
        # 16x16 -> 8x8
        self.c5 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.p5 = nn.MaxPool2d(2, stride=2)


        self.lstm = nn.LSTM(256, self.hidden_size, self.num_layers, batch_first=True)
        
        # self.f = nn.Flatten()
        # self.l = nn.Linear(32*32, 256) 
        
        if use_cuda:
            self.cuda(device=device)
        
    def forward(self, data):
        device = self.device
        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid
        hidden_size =self.hidden_size
        num_layers = self.num_layers
        lstm = self.lstm
        n = data.size(0)
        
        h = self.c1(data)
        h = tanh(h)
        h = self.p1(h)
        
        h = self.c2(h)
        h = tanh(h)
        h = self.p2(h)
        
        h = self.c3(h)
        h = tanh(h)
        h = self.p3(h)
        
        h = self.c4(h)
        h = tanh(h)
        h = self.p4(h)
        
        h = self.c5(h)
        h = tanh(h)
        h = self.p5(h)
        
        # h = self.f(h)
        # h = self.l(h)


        # # h: (N, 256, 8, 8)
        # # seqence length
        # l = n_grid_height*n_grid_width
        # h = torch.permute(h, (0, 2, 3, 1)).view(n, l, -1)

        # hiddens = torch.randn(num_layers, n, hidden_size).to(device=device)
        # states = torch.randn(num_layers, n, hidden_size).to(device=device)

        # outputs = torch.zeros(n, l, hidden_size, maximum_in_grid).to(device=device)
        # for i in range(maximum_in_grid):
        #     output, (hiddens, states) = lstm(h, (hiddens, states))
        #     outputs[..., i] = output

        # h = outputs
        # h = h.view(n, n_grid_height, n_grid_width, hidden_size, maximum_in_grid)
        # h = torch.permute(h, (0, 3, 1, 2, 4))

        # h: (N, 256, 8, 8)
        # seqence length
        l = n_grid_height*n_grid_width
        h = torch.permute(h, (0, 2, 3, 1)).reshape(n*l, 1, -1)

        hiddens = torch.zeros((num_layers, n*l, hidden_size), dtype=h.dtype).to(device=device)
        states = torch.zeros((num_layers, n*l, hidden_size), dtype=h.dtype).to(device=device)

        outputs = torch.zeros(n*l, 1, hidden_size, maximum_in_grid).to(device=device)
        for i in range(maximum_in_grid):
            _, (hiddens, states) = lstm(h, (hiddens, states))
            outputs[:, 0, :, i] = hiddens[num_layers-1]

        h = outputs
        h = h.view(n, n_grid_height, n_grid_width, hidden_size, maximum_in_grid)
        h = torch.permute(h, (0, 3, 1, 2, 4))

        h = tanh(h)

        # h = h[..., None]
        
        return h
        
class PredictSPAIR(nn.Module):
    def __init__(self, use_cuda=True,device=None, 
        height=256, width=256, 
        n_grid_height=8, n_grid_width=8, maximum_in_grid=3, 
        *args, **kargs):
        super().__init__(*args, **kargs)
        
        self.l1 = nn.Linear(256, 256*256)
        
        # 256x256 -> 128x128
        self.c1 = nn.Conv2d(1, 4, 3, 1, padding="same")
        self.p1 = nn.MaxPool2d(2, stride=2)
        
        # 128x128 -> 64x64
        self.c2 = nn.Conv2d(4, 8, 3, 1, padding="same")
        self.p2 = nn.MaxPool2d(2, stride=2)
        
        # 64x64 -> 32x32
        self.c3 = nn.Conv2d(8, 16, 3, 1, padding="same")
        self.p3 = nn.MaxPool2d(2, stride=2)
        
        # 32x32 -> 16x16
        self.c4 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.p4 = nn.MaxPool2d(2, stride=2)
        
        # 16x16 -> 8x8
        self.c5 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.p5 = nn.MaxPool2d(2, stride=2)
        
        self.f = nn.Flatten()
        self.l = nn.Linear(64*8*8, 9)
        
        if use_cuda:
            self.cuda(device=device)
        
    def forward(self, compart_vector):
        h = self.l1(compart_vector).view(-1, 1, 256, 256)
        # h = torch.cat([h, processed], 1)
        
        h = self.c1(h)
        h = tanh(h)
        h = self.p1(h)
        
        h = self.c2(h)
        h = tanh(h)
        h = self.p2(h)
        
        h = self.c3(h)
        h = tanh(h)
        h = self.p3(h)
        
        h = self.c4(h)
        h = tanh(h)
        h = self.p4(h)
        
        h = self.c5(h)
        h = tanh(h)
        h = self.p5(h)
        
        h = self.f(h)
        a = self.l(h)
        
        z_pres_p = sigmoid(a[:, 0:1]) # Squish to [0,1]
        z_where_loc = a[:, 1:5]
        z_where_scale = softplus(a[:, 5:]) # Squish to >0
        return z_pres_p, z_where_loc, z_where_scale


class BaselineStep(nn.Module):
    def __init__(self, use_cuda=True, device=None, 
        height=256, width=256, 
        n_grid_height=8, n_grid_width=8, maximum_in_grid=3, 
        *args, **kargs):
        super().__init__(*args, **kargs)
        
        self.l1 = nn.Linear(256, 256*256)
        
        # 256x256 -> 128x128
        self.c1 = nn.Conv2d(1, 4, 3, 1, padding="same")
        self.p1 = nn.MaxPool2d(2, stride=2)
        
        # 128x128 -> 64x64
        self.c2 = nn.Conv2d(4, 8, 3, 1, padding="same")
        self.p2 = nn.MaxPool2d(2, stride=2)
        
        # 64x64 -> 32x32
        self.c3 = nn.Conv2d(8, 16, 3, 1, padding="same")
        self.p3 = nn.MaxPool2d(2, stride=2)
        
        # 32x32 -> 16x16
        self.c4 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.p4 = nn.MaxPool2d(2, stride=2)
        
        # 16x16 -> 8x8
        self.c5 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.p5 = nn.MaxPool2d(2, stride=2)
        
        self.f = nn.Flatten()
        self.l2 = nn.Linear(64*8*8, 1)
        
        if use_cuda:
            self.cuda(device=device)
    
    def forward(self, compart_vector):
        
        h = self.l1(compart_vector.detach()).view(-1, 1, 256, 256)
        # h = torch.cat([h, processed_objects.detach()], 1)
        
        h = self.c1(h)
        h = tanh(h)
        h = self.p1(h)
        
        h = self.c2(h)
        h = tanh(h)
        h = self.p2(h)
        
        h = self.c3(h)
        h = tanh(h)
        h = self.p3(h)
        
        h = self.c4(h)
        h = tanh(h)
        h = self.p4(h)
        
        h = self.c5(h)
        h = tanh(h)
        h = self.p5(h)
        
        
        # h = self.l2(h.view(-1, 16*32*32))
        h = self.f(h)
        h = self.l2(h)
        
        return h