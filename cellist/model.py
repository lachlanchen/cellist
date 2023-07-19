import os
from collections import namedtuple
import pyro
import pyro.optim as optim
# from pyro.infer import SVI, TraceGraph_ELBO
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.contrib.examples.multi_mnist as multi_mnist
import torch
import torch.nn as nn
from torch.nn.functional import relu, tanh, sigmoid, softplus, grid_sample, affine_grid
import numpy as np

smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.0')


device = torch.device('cuda:0')


X_np = torch.load("X").numpy()

N, img_height, img_width = X_np.shape
obj_height, obj_width = 100, 100

bounding_box_height_min, bounding_box_width_min = 30, 30
bounding_box_height_max, bounding_box_width_max = 70, 70

bounding_box_height_scale_min, bounding_box_width_scale_min = \
    img_height / bounding_box_height_max, img_width / bounding_box_width_max
bounding_box_height_scale_max, bounding_box_width_scale_max = \
    img_height / bounding_box_height_min, img_width / bounding_box_width_min

bouding_box_scale_min = bounding_box_width_scale_min
bouding_box_scale_max = bounding_box_width_scale_max

z_what_latent_size = 100


z_where_prior_loc = torch.tensor([5., 0., 0., 0.])
z_where_prior_scale = torch.tensor([10, 1, 1, 1])
z_what_prior_loc = torch.zeros(z_what_latent_size)
z_what_prior_scale = torch.ones(z_what_latent_size)

s = 0.5
z_pres_counts_p_ = torch.tensor([s * ((1-s)**k) / (1-(1-s)**(64+1)) for k in range(65)]).view(1, 65) #+ 1.4013e-45
print(z_pres_counts_p_)



def expand_z_where(z_where):
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [sx, sy, x, y] -> [[sx,0,x],
    #                    [0,sy,y]]
    n = z_where.size(0)
    expansion_indices = torch.cuda.LongTensor([1, 0, 3, 0, 2, 4], device=device)
    out = torch.cat((torch.zeros([1, 1]).expand(n, 1).cuda(device=device), z_where), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

def object_to_image(z_where, obj):
    n = obj.size(0)
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n, 1,img_height, img_width)))
    out = grid_sample(obj.view(n, 1, obj_height, obj_width), grid)
    return out.view(n, img_height, img_width)

# def object_to_image_in_compartment(i, j, z_where, obj):
#     n = obj.size(0)
#     theta = expand_z_where(z_where)
#     grid = affine_grid(theta, torch.Size((n, 1,img_height, img_width)))
#     out = grid_sample(obj.view(n, 1, obj_height, obj_width), grid)
#     return out.view(n, img_height, img_width)

def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat(
        (
            torch.ones([1, 1], device=device).type_as(z_where).expand(n, 1) / z_where[:, 0:1], 
            torch.ones([1, 1], device=device).type_as(z_where).expand(n, 1) / z_where[:, 1:2], 
            -z_where[:, 2:3] / z_where[:, 0:1],
            -z_where[:, 3:4] / z_where[:, 1:2]
        ), 1)
    # out = out / z_where[:, 0:1]
    return out

def image_to_object(z_where, image):
    n = image.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv, torch.Size((n, 1, obj_height, obj_width)))
    out = grid_sample(image.view(n, 1, img_height, img_width), grid)
    return out.view(n, -1)

def image_to_selected(z_where, image):
    n = image.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv, torch.Size((n, 1, obj_height, obj_width)))
    out = grid_sample(image.view(n, 1, img_height, img_width), grid)
    
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n ,1, img_height, img_width)))
    out = grid_sample(out.view(n, 1, obj_height, obj_width), grid)
    
    # return out.view(n, -1)
    return out


# Create the neural network. This takes a latent code, z_what, to pixel intensities.
class Decoder(nn.Module):
    def __init__(self, use_cuda=True, *args, **kargs):
        super().__init__(*args, **kargs)
        
        self.l1 = nn.Linear(z_what_latent_size, 1024*64)
        self.u1 = nn.Unflatten(1, (1024, 8, 8))
        
        # 8x8 -> 16x16
        self.t1  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c1a = nn.Conv2d(512, 256, 3, padding="same")
        self.c1b = nn.Conv2d(256, 128, 3, padding="same")
        self.c1c = nn.Conv2d(128, 64, 3, padding="same")
        self.c1d = nn.Conv2d(64, 32, 3, padding="same")
        
        # 16x16 -> 32x32
        self.t2  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.c2a = nn.Conv2d(16, 8, 3, padding="same")
        self.c2b = nn.Conv2d(8, 4, 3, padding="same")
        self.c2c = nn.Conv2d(4, 2, 3, padding="same")
        self.c2d = nn.Conv2d(2, 1, 3, padding="same")
        
        
        self.f2 = nn.Flatten()
        self.l2 = nn.Linear(32*32, obj_height*obj_width)
        
        if use_cuda:
            self.cuda(device=device)

    def forward(self, z_what):
        # h = relu(self.l1(z_what))
        # h = self.l1(z_what)
        # h = relu(h)
        # h = self.l2(h)
        # h = sigmoid(h)
        
        h = self.l1(z_what)
        h = tanh(h)
        h = self.u1(h)
        
        h = self.t1(h)
        h = tanh(h)
        h = self.c1a(h)
        h = tanh(h)
        h = self.c1b(h)
        h = tanh(h)
        h = self.c1c(h)
        h = tanh(h)
        h = self.c1d(h)
        h = tanh(h)
        
        h = self.t2(h)
        h = tanh(h)
        h = self.c2a(h)
        h = tanh(h)
        h = self.c2b(h)
        h = tanh(h)
        h = self.c2c(h)
        h = tanh(h)
        h = self.c2d(h)
        h = tanh(h)
        
        h = self.f2(h)
        h = tanh(h)
        h = self.l2(h)
        
        
        h = sigmoid(h)
        
        # return sigmoid(self.l2(h))
        return h

# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
class Encoder(nn.Module):
    def __init__(self, use_cuda=True, *args, **kargs):
        super().__init__(*args, **kargs)
        
        # self.l1 = nn.Linear(obj_height*obj_width, 200)
        # self.l2 = nn.Linear(200, z_what_latent_size*2)
        
        self.l1 = nn.Linear(obj_height*obj_width, 32*32)
        self.u1 = nn.Unflatten(1, (1, 32, 32))
        
        # 32x32 -> 16x16
        self.c1 = nn.Conv2d(1, 2, 3, 1, padding="same")
        self.p1 = nn.MaxPool2d(2, stride=2)
        self.c1a = nn.Conv2d(2, 4, 3, 1, padding="same")
        self.c1b = nn.Conv2d(4, 8, 3, 1, padding="same")
        self.c1c = nn.Conv2d(8, 16, 3, 1, padding="same")
        
        # 16x16 -> 8x8
        self.c2 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.p2 = nn.MaxPool2d(2, stride=2)
        self.c2a = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.c2b = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.c2c = nn.Conv2d(128, 256, 3, 1, padding="same")
        
        self.f2 = nn.Flatten()
        self.l2 = nn.Linear(256*8*8, z_what_latent_size*2)
        
        
        if use_cuda:
            self.cuda(device=device)

    def forward(self, data):
        # h = relu(self.l1(data))
        # h = self.l1(data)
        # h = relu(h)
        # a = self.l2(h)
        
        h = self.l1(data)
        h = tanh(h)
        h = self.u1(h)
        
        h = self.c1(h)
        h = tanh(h)
        h = self.p1(h)
        h = self.c1a(h)
        h = tanh(h)
        h = self.c1b(h)
        h = tanh(h)
        h = self.c1c(h)
        h = tanh(h)
        
        h = self.c2(h)
        h = tanh(h)
        h = self.p2(h)
        h = self.c2a(h)
        h = tanh(h)
        h = self.c2b(h)
        h = tanh(h)
        h = self.c2c(h)
        h = tanh(h)
        
        h = self.f2(h)
        a = self.l2(h)
        
        z_loc = a[:, 0:z_what_latent_size]
        z_scale = softplus(a[:, z_what_latent_size:])
        return z_loc, z_scale
        
## 一个 compartment 一个细胞就不需要 LSTM
class Compartment(nn.Module):
    def __init__(self ,use_cuda=True, *args, **kargs):
        super().__init__(*args, **kargs)
        
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
        
        # self.f = nn.Flatten()
        # self.l = nn.Linear(32*32, 256) 
        
        if use_cuda:
            self.cuda(device=device)
        
    def forward(self, data):
        
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
        
        return h
        
class PredictSPAIR(nn.Module):
    def __init__(self ,use_cuda=True, *args, **kargs):
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
        
    def forward(self, compart_vector, processed):
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
    def __init__(self):
        super().__init__()
        
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
        
        # self.cuda(device=device)
    
    def forward(self, compart_vector, processed_objects):
        
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

      


# def prior_step(n, t, prev_x, prev_z_pres):
def prior_step(n, i, j, background_left, prev_x, prev_z_pres, z_pres_p_unmasked):

    # Sample variable indicating whether to add this object to the output.

    # We multiply the success probability of 0.5 by the value sampled for this
    # choice in the previous step. By doing so we add objects to the output until
    # the first 0 is sampled, after which we add no further objects.
    
    k = i*8+j+1
    # z_pres_counts_p = z_pres_counts_p_.expand(n, 65)
    # z_pres_p = torch.zeros([n, 65])
    counts = torch.arange(0, 65).view(1, 65).expand(n, -1)
    prev_z_pres_counts = prev_z_pres.expand(n, 65)
    z_pres_p = z_pres_p_unmasked * torch.clamp((counts - prev_z_pres_counts) * torch.gt(counts, prev_z_pres_counts) / (64 - (k-1)), min=0, max=1)
    
    # print(z_pres_p.sum(1, keepdim=True))
    
    
    # print("k: ", k)
    # print("denominator: ", (counts - prev_z_pres_counts) * torch.gt(counts, prev_z_pres_counts) / (64 - (k-1)) )
    # print("counts: ", counts)
    # print("prev_z_pres_counts: ", prev_z_pres_counts)
    # print("z_pres_p.shape: ", z_pres_p.shape)
    # print("z_pres_p: ", z_pres_p.sum(1))
    
    z_pres = pyro.sample('z_pres_{}_{}'.format(i, j),
                         dist.Bernoulli(z_pres_p.sum(1, keepdim=True).expand(n, -1))
                             .to_event(1)).cuda(device=device)

    z_where = pyro.sample('z_where_{}_{}'.format(i, j),
                          dist.Normal(z_where_prior_loc.expand(n, -1),
                                      z_where_prior_scale.expand(n, -1))
                              .mask(z_pres.cpu())
                              .to_event(1)).cuda(device=device)

    z_what = pyro.sample('z_what_{}_{}'.format(i, j),
                         dist.Normal(z_what_prior_loc.expand(n, -1),
                                     z_what_prior_scale.expand(n, -1))
                             .mask(z_pres.cpu())
                             .to_event(1)).cuda(device=device)
    
    z_where_resize_x = sigmoid(z_where[:, 0:1])*(bouding_box_scale_max - bouding_box_scale_min)+bouding_box_scale_min
    z_where_resize_y_over_x = sigmoid(z_where[:, 1:2]) * (1.25-0.75) + 0.75
    z_where_resize_y = z_where_resize_x * z_where_resize_y_over_x
    z_where_rescaled = torch.cat(
        [
            z_where_resize_x, 
            z_where_resize_x * z_where_resize_y_over_x,
            z_where_resize_x * (tanh(z_where[:,2:3]) * 0.5 * 0.25 - (j-4+0.5)/4.),
            z_where_resize_y * (tanh(z_where[:,3:4]) * 0.5 * 0.25 - (i-4+0.5)/4.)
        ], 
        1
    )
    
    z_where_in_compartment = z_where_rescaled

    y_attend = decode(z_what)
    y = object_to_image(z_where_in_compartment, y_attend)
    
    background_left_last = background_left
    object_visible = image_to_object(z_where_in_compartment, background_left)
    object_score = torch.clamp(object_to_image(z_where_in_compartment, object_visible), 0., 1.)
    background_left = background_left * (1. - object_score * z_pres.view(-1, 1, 1))


    # Combine the image generated at this step with the image so far.
    x = prev_x + y * background_left_last * z_pres.view(-1, 1, 1)


    return x, background_left, z_where_in_compartment, prev_z_pres + z_pres.cpu(), z_pres_p


def prior(n):
    x = torch.zeros(n, img_height, img_width).cuda(device=device)
    z_wheres = []
    z_preses = []
    
    z_pres = torch.zeros(n, 1)
    z_pres_p = z_pres_counts_p_.expand(n, -1)
    background_left_last = background_left = torch.ones(n, img_height, img_width).cuda(device=device)
    for i in range(8):
        for j in range(8):
            
            x, background_left, z_where, z_pres, z_pres_p = prior_step(n, i, j, background_left, x, z_pres, z_pres_p)
            z_wheres.append(z_where)
            z_preses.append(z_pres)
    
    return x, z_wheres, z_preses



def model(data, y_pres):
    # Register network for optimization.
    pyro.module("decode", decode)
    with pyro.plate('data', data.size(0)) as indices:
        batch = data[indices]
        x, z_wheres, z_preses = prior(batch.size(0))
        x = x.view(-1, img_height * img_width).cpu()
        # x = prior(batch.size(0))
        sd = (0.3 * torch.ones(1)).expand_as(x)
        pyro.sample('obs', dist.Normal(x, sd).to_event(1),
                    obs=batch.cpu())
        
        comp_v = compartment(batch.view(-1, 1, img_height, img_width).cuda(device=device))
        for i in range(8):
            for j in range(8):
                y_pres_p, _, _ = predict_spair(comp_v[:, :, i, j], None)
                
                with pyro.poutine.scale(scale=46):
                    pyro.sample(f"y_pres_{i}_{j}", dist.Bernoulli(y_pres_p.cpu()).to_event(1), obs=y_pres[indices,i,j].view(-1, 1).cpu())


def guide_step(n, i, j, data, y_pres, compartment_vector, prev_processed):

    z_pres_p, z_where_loc, z_where_scale = predict_spair(compartment_vector, prev_processed)

    # Here we compute the baseline value, and pass it to sample.
    baseline_value = baseline_step(compartment_vector, prev_processed).cpu()
    
    z_pres = pyro.sample('z_pres_{}_{}'.format(i, j),
                         dist.Bernoulli(z_pres_p.cpu())
                             .to_event(1),
                         infer=dict(baseline=dict(baseline_value=baseline_value.squeeze(-1)))).cuda(device=device)

    z_where = pyro.sample('z_where_{}_{}'.format(i, j),
                          dist.Normal(z_where_loc.cpu(), z_where_scale.cpu())
                              .mask(z_pres.cpu())
                              .to_event(1)).cuda(device=device)
    
    z_where_resize_x = sigmoid(z_where[:, 0:1])*(bouding_box_scale_max - bouding_box_scale_min)+bouding_box_scale_min
    z_where_resize_y_over_x = sigmoid(z_where[:, 1:2]) * (1.25-0.75) + 0.75
    z_where_resize_y = z_where_resize_x * z_where_resize_y_over_x
    z_where_rescaled = torch.cat(
        [
            z_where_resize_x, 
            z_where_resize_x * z_where_resize_y_over_x,
            z_where_resize_x * (tanh(z_where[:,2:3]) * 0.5 * 0.25 - (j-4+0.5)/4.),
            z_where_resize_y * (tanh(z_where[:,3:4]) * 0.5 * 0.25 - (i-4+0.5)/4.)
        ], 
        1
    )
    
    z_where_in_compartment = z_where_rescaled

    
    x_attend = image_to_object(z_where_in_compartment, data)

    z_what_loc, z_what_scale = encode(x_attend)

    z_what = pyro.sample('z_what_{}_{}'.format(i, j),
                         dist.Normal(z_what_loc.cpu(), z_what_scale.cpu())
                             .mask(z_pres.cpu())
                             .to_event(1)).cuda(device=device)
    

    return z_pres, z_where_in_compartment, image_to_selected(z_where_in_compartment, data)*z_pres.view(-1, 1, 1, 1) + prev_processed

def guide(data, y_pres):
    # Register networks for optimization.
    pyro.module("compartment", compartment)
    pyro.module("predict_spair", predict_spair)
    pyro.module("encode", encode)
    pyro.module("baseline_step", baseline_step)

    with pyro.plate('data', data.size(0), subsample_size=4) as indices:
        n = indices.size(0)
        
        batch = data[indices].view(-1, 1, img_height, img_width).cuda(device=device)
        compartment_vector = compartment(batch)
        processed = torch.zeros_like(batch).cuda(device=device)
        
        steps = []
        for i in range(8):
            for j in range(8):
                z_pres, z_where, processed = guide_step(n, i, j, batch, y_pres[indices], compartment_vector[:, :, i, j], processed)
                # print(processed)
                steps.append((z_pres, z_where, processed))
        return steps


decode=Decoder(use_cuda=True)
encode = Encoder().cuda(device=device)
compartment = Compartment().cuda(device=device)
predict_spair = PredictSPAIR().cuda(device=device)
baseline_step = BaselineStep().cuda(device=device)                    

optimizer = optim.Adam({'lr': 1e-4})


model_root = f"model_ss_0.5_rectangle_best"
def load_model(model_root):
    encode_path = os.path.join(model_root, "encode.pt")
    decode_path = os.path.join(model_root, "decode.pt")
    compartment_path = os.path.join(model_root, "compartment.pt")
    predict_spair_path = os.path.join(model_root, "predict_spair.pt")
    baseline_step_path = os.path.join(model_root, "baseline_step.pt")
    optim_path = os.path.join(model_root, "optim.pt")
    model_path = os.path.join(model_root, "mymodel.pt")
    
    encode.load_state_dict(torch.load(encode_path))
    decode.load_state_dict(torch.load(decode_path))
    compartment.load_state_dict(torch.load(compartment_path))
    predict_spair.load_state_dict(torch.load(predict_spair_path))
    baseline_step.load_state_dict(torch.load(baseline_step_path))

    optimizer.load(optim_path)

    pyro.get_param_store().load(model_path)


load_model(model_root)