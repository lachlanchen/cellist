import os
from collections import namedtuple, OrderedDict
import numpy as np
import pandas as pd
from skimage import io
import cv2
from sqlalchemy import inspect, create_engine

import pyro
import pyro.optim as optim
# from pyro.infer import SVI, TraceGraph_ELBO
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.contrib.examples.multi_mnist as multi_mnist
import torch
import torch.nn as nn
from torch.nn.functional import relu, tanh, sigmoid, softplus, grid_sample, affine_grid



from cellist.dicton import Dicton
from cellist.model_2d_components import Decoder, Encoder, Compartment, PredictSPAIR, BaselineStep
from cellist.model_2d_utilities import *
# from cellist.model_2d_constants import *
from cellist.utils.constants import curdir, dataroot, mysqlconfig, mysqlurl

import json
import shutil

# smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.0')

model_home = "data/models"

# torch.set_default_dtype(torch.float32)
default_dtype = torch.float32

class ModelD2Init(Dicton):


    def __del__(self):
        pass

    def __init__(self, model_id, lr=1e-4, *args, **kargs):
        if self.returned_exist == True:
            print("Returned instance without initialization. ")
            return

        self.init_constants(model_id, lr=lr, *args, **kargs)

        # pyro.set_rng_seed(0)

        self.generate_dataset()
        self.init_components()


    def init_constants(self, model_id, lr=1e-4):
        self.model_id = model_id

        img_height, img_width, object_min, object_max, n_grid_height, n_grid_width, maximum_in_grid = \
            get_model_info(model_id)

        # provided by user
        self.img_height, self.img_width = 256, 256
        self.n_grid_height, self.n_grid_width = 8, 8
        self.maximum_in_grid = maximum_in_grid
        self.obj_min, self.obj_max = object_min, object_max

        self.obj_height, self.obj_width = 100, 100
        self.z_what_latent_size = 1000

        # model hyperparameters
        self.device = torch.device('cuda:0')
        self.lr = lr
        self.aux_loss_multiplier = 0.5
        # self.aux_loss_multiplier = 46
        self.subsample_size = 1
        self.current_epoch = 0
        # self.loss = float("inf")
        self.loss = float(np.finfo(np.float32).max)



        # derived parameters
        bounding_box_height_min, bounding_box_width_min = self.obj_min, self.obj_min
        bounding_box_height_max, bounding_box_width_max = self.obj_max, self.obj_max

        bounding_box_height_scale_min, bounding_box_width_scale_min = \
            self.img_height / bounding_box_height_max, self.img_width / bounding_box_width_max
        bounding_box_height_scale_max, bounding_box_width_scale_max = \
            self.img_height / bounding_box_height_min, self.img_width / bounding_box_width_min

        self.bounding_box_scale_min = bounding_box_scale_min = bounding_box_height_scale_min
        self.bounding_box_scale_max = bounding_box_scale_max = bounding_box_height_scale_max


        sx_loc = sy_loc = (bounding_box_scale_min + bounding_box_scale_max)/2
        sx_scale = sy_scale = (bounding_box_scale_max - bounding_box_scale_min)*2

        print("sx_loc: ", sx_loc)
        print("sx_scale: ", sx_scale)

        
        # prior
        self.z_where_prior_loc = torch.tensor([sx_loc, sx_loc, 0., 0.])
        self.z_where_prior_scale = torch.tensor([sx_scale, sy_scale, 1, 1])
        self.z_what_prior_loc = torch.zeros(self.z_what_latent_size)
        self.z_what_prior_scale = torch.ones(self.z_what_latent_size)

        self.n_grids = self.n_grid_height * self.n_grid_width * self.maximum_in_grid
        n_counts = self.n_grids + 1
        self.s = s = 0.1
        self.z_pres_counts_prior = torch.tensor([s * ((1-s)**k) / (1-(1-s)**(self.n_grids+1)) for k in range(n_counts)]).view(1, n_counts) #+ 1.4013e-45
        print("z_pres_counts_prior: \n", self.z_pres_counts_prior)


        self.model_root = model_root = os.path.join(model_home, model_id)
        os.makedirs(self.model_root, exist_ok=True)
        


        self.encode_path = os.path.join(model_root, "encode.pt")
        self.decode_path = os.path.join(model_root, "decode.pt")
        self.compartment_path = os.path.join(model_root, "compartment.pt")
        self.predict_spair_path = os.path.join(model_root, "predict_spair.pt")
        self.baseline_step_path = os.path.join(model_root, "baseline_step.pt")
        self.optim_path = os.path.join(model_root, "optim.pt")
        self.model_path = os.path.join(model_root, "mymodel.pt")
        self.config_path = os.path.join(model_root, f"{self.__class__.__name__}-config.json")

        # self.encode = None
        # self.decode = None
        # self.compartment = None
        # self.predict_spair = None
        # self.baseline_step = None
        # self.encode = None
        # self.optim = None
        # self.model = None
        # self.config = None

        self.training = False
        self.enable_train = True
        self.loaded = False
        self.dataset = None
        self.dataset_min = None
        self.dataset_max = None

        # self.save_table = ["encode", "decode", "compartment", "predict_spair", "baseline_step", "model", "optimizer", "config"]
        self.save_table = ["compartment", "predict_spair", "baseline_step", "model", "optimizer", "config"]
        # self.save_table = {
        #     "encode": {

        #     }
        # }

        self.max_epoch = 50000

    def init_components(self):
        print("Assembling...")
        device = self.device
        lr = self.lr

        z_what_latent_size = self.z_what_latent_size
        img_height = self.img_height
        img_width = self.img_width
        obj_height = self.obj_height
        obj_width = self.obj_width
        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid

        self.decode = Decoder(
            use_cuda=True, device=device, 
            z_what_latent_size=z_what_latent_size, 
            obj_height=obj_height, obj_width=obj_width
        )
        self.encode = Encoder(
            use_cuda=True, device=device, 
            z_what_latent_size=z_what_latent_size, 
            obj_height=obj_height, obj_width=obj_width
        )
        self.compartment = Compartment(
            use_cuda=True, device=device, 
            height=img_height, width=img_width, 
            n_grid_height=n_grid_height, n_grid_width=n_grid_width, 
            maximum_in_grid=maximum_in_grid
        )
        self.predict_spair = PredictSPAIR(
            use_cuda=True, device=device,
            height=img_height, width=img_width, 
            n_grid_height=n_grid_height, n_grid_width=n_grid_width, 
            maximum_in_grid=maximum_in_grid
        )
        self.baseline_step = BaselineStep(
            use_cuda=True, device=device,
            height=img_height, width=img_width, 
            n_grid_height=n_grid_height, n_grid_width=n_grid_width, 
            maximum_in_grid=maximum_in_grid
        ) 
        self.optimizer = optim.Adam({'lr': lr})
        self.param_store = pyro.get_param_store()

        self.decode.train(False)
        self.encode.train(False)
    

        self.gen_svi_model()

        self.load_svi()
        self.save_svi()

        print("Assembled.")

    def prior_step_lstm(
        self, 
        n, i, j, t, 
        background_left, prev_x, 
        prev_z_pres_single_in_compartment, prev_z_pres, 
        z_pres_p_unmasked):

        device = self.device
        decode = self.decode

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid

        z_where_prior_loc = self.z_where_prior_loc
        z_where_prior_scale = self.z_where_prior_scale
        z_what_prior_loc = self.z_what_prior_loc
        z_what_prior_scale = self.z_what_prior_scale 

        n_grids = self.n_grids
        n_counts = n_grids + 1

        n_grid_height_half = n_grid_height / 2.
        n_grid_width_half = n_grid_width / 2.

        # n_grids = n_grid_height * n_grid_width * maximum_in_grid
        

        bounding_box_scale_min = self.bounding_box_scale_min
        bounding_box_scale_max = self.bounding_box_scale_max
        
        # Sample variable indicating whether to add this object to the output.
        k = (i*n_grid_width+j)*maximum_in_grid + t +1
        counts = torch.arange(0, n_counts).view(1, n_counts).expand(n, -1)
        prev_z_pres_counts = prev_z_pres.expand(n, n_counts)
        z_pres_p = z_pres_p_unmasked * torch.clamp((counts - prev_z_pres_counts) * torch.gt(counts, prev_z_pres_counts) / (n_counts - (k-1)), min=0, max=1)
        
        
        z_pres = pyro.sample('z_pres_{}_{}_{}'.format(i, j, t),
                             dist.Bernoulli(z_pres_p.sum(1, keepdim=True).expand(n, -1) * prev_z_pres_single_in_compartment)
                                 .to_event(1)).cuda(device=device)

        z_where = pyro.sample('z_where_{}_{}_{}'.format(i, j, t),
                              dist.Normal(z_where_prior_loc.expand(n, -1),
                                          z_where_prior_scale.expand(n, -1))
                                  .mask(z_pres.cpu())
                                  .to_event(1)).cuda(device=device)

        z_what = pyro.sample('z_what_{}_{}_{}'.format(i, j, t),
                             dist.Normal(z_what_prior_loc.expand(n, -1),
                                         z_what_prior_scale.expand(n, -1))
                                 .mask(z_pres.cpu())
                                 .to_event(1)).cuda(device=device)
        
        z_where_resize_x = sigmoid(z_where[:, 0:1])*(bounding_box_scale_max - bounding_box_scale_min)+bounding_box_scale_min
        z_where_resize_y_over_x = sigmoid(z_where[:, 1:2]) * (1.25-0.75) + 0.75
        z_where_resize_y = z_where_resize_x * z_where_resize_y_over_x
        z_where_rescaled = torch.cat(
            [
                z_where_resize_x, 
                z_where_resize_x * z_where_resize_y_over_x,
                z_where_resize_x * (tanh(z_where[:,2:3]) * 0.5 / n_grid_width_half  - (j-n_grid_width_half+0.5)/n_grid_width_half),
                z_where_resize_y * (tanh(z_where[:,3:4]) * 0.5 / n_grid_height_half - (i-n_grid_height_half+0.5)/n_grid_height_half)
            ], 
            1
        )
        
        z_where_in_compartment = z_where_rescaled

        y_attend, _ = decode(z_what)
        y = object_to_image(z_where_in_compartment, y_attend)
        
        background_left_last = background_left
        object_visible = image_to_object(z_where_in_compartment, background_left)
        object_score = torch.clamp(object_to_image(z_where_in_compartment, object_visible), 0., 1.)
        background_left = background_left * (1. - object_score * z_pres.view(-1, 1, 1))


        # Combine the image generated at this step with the image so far.
        x = prev_x + y * background_left_last * z_pres.view(-1, 1, 1)


        # return x, background_left, z_where_in_compartment, prev_z_pres + z_pres.cpu(), z_pres_p
        return x, background_left, z_pres.cpu(), z_pres_p


    def prior_step(
        self, 
        n, i, j, 
        background_left, prev_x, 
        prev_z_pres, z_pres_p_unmasked):

        prev_z_pres_in_compartment = 1
        z_pres_total_in_compartment = 0
        for t in range(self.maximum_in_grid):
            x, background_left, prev_z_pres_in_compartment, z_pres_p = \
                self.prior_step_lstm(
                    n, i, j, t, 
                    background_left, prev_x, 
                    prev_z_pres_in_compartment, prev_z_pres+z_pres_total_in_compartment, 
                    z_pres_p_unmasked
                )
            z_pres_total_in_compartment += prev_z_pres_in_compartment

        return x, background_left, z_pres_total_in_compartment, z_pres_p

    def prior(self, n):

        img_height = self.img_height
        img_width = self.img_width

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        device = self.device
        prior_step = self.prior_step

        z_pres_counts_prior = self.z_pres_counts_prior


        x = torch.zeros(n, img_height, img_width).cuda(device=device)
        z_wheres = []
        z_preses = []
        
        prev_z_pres = torch.zeros(n, 1)
        z_pres_p = z_pres_counts_prior.expand(n, -1)
        background_left_last = background_left = torch.ones(n, img_height, img_width).cuda(device=device)
        for i in range(n_grid_height):
            for j in range(n_grid_width):
                
                # x, background_left, z_pres, z_pres_p = prior_step(n, i, j, background_left, x, prev_z_pres, z_pres_p)
                x, background_left, z_pres, z_pres_p = prior_step(n, i, j, background_left, x, prev_z_pres, z_pres_p)

                prev_z_pres = prev_z_pres + z_pres
                # z_wheres.append(z_where)
                # z_preses.append(z_pres)
        
        # z_wheres = torch.stack(z_wheres, axis=0)
        # z_preses = torch.stack(z_preses, axis=0)

        # z_wheres = torch.swapaxes(z_wheres, 0, 1)
        # z_preses = torch.swapaxes(z_preses, 0, 1)


        # return x, z_wheres, z_preses
        return x


    def model(self, dataset):
        device = self.device

        decode = self.decode
        compartment = self.compartment
        predict_spair = self.predict_spair

        prior = self.prior

        img_height = self.img_height
        img_width = self.img_width
        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width

        # images, y_pres = dataset
        images = dataset["images"]
        # y_pres_algorithm = dataset["y_pres_algorithm"]
        # y_pres_manual = dataset["y_pres_manual"]
        y_pres = dataset["y_pres"]

        # Register network for optimization.
        pyro.module("decode", decode)
        pyro.module("compartment", compartment)
        pyro.module("predict_spair", predict_spair)
        with pyro.plate('images', images.size(0)) as indices:
            batch = images[indices]
            # x, z_wheres, z_preses = prior(batch.size(0))
            x = prior(batch.size(0))
            x = x.view(-1, img_height * img_width).cpu()

            # x = prior(batch.size(0))
            sd = (0.3 * torch.ones(1)).expand_as(x)
            pyro.sample(
                'obs', 
                dist.Normal(x, sd).to_event(1),
                obs=batch.cpu()
            )




            comp_v = compartment(batch.view(-1, 1, img_height, img_width).cuda(device=device))
            # for batch_idx, dataset_idx in enumerate(indices):

            #     if y_pres[dataset_idx] == None:
            #         continue

            #     for i in range(n_grid_height):
            #         for j in range(n_grid_width):

            #             if type(y_pres) is np.ndarray:
            #                 y_pres_single = torch.from_numpy(y_pres[dataset_idx,i,j]).view(-1, 1).cpu()
            #             else:
            #                 y_pres_single = y_pres[dataset_idx,i,j].view(-1, 1).cpu()

            #             y_pres_p, _, _ = predict_spair(comp_v[batch_idx:batch_idx+1, :, i, j], None)
                        
            #             with pyro.poutine.scale(scale=46):
            #                 pyro.sample(f"y_pres_{i}_{j}", dist.Bernoulli(y_pres_p.cpu()).to_event(1), obs=y_pres_single)

            for i in range(n_grid_height):
                for j in range(n_grid_width):
                    # if type(y_pres) is np.ndarray:
                    #     y_pres_batch = torch.from_numpy(y_pres[indices, i, j]).view(-1, 1).cpu()
                    # else:
                    #     y_pres_batch = y_pres[indices, i, j].view(-1, 1).cpu()

                    y_pres_batch = y_pres[indices, i, j].view(-1, 1).cpu()

                    if y_pres_batch.min() == -1:
                        continue

                    for t in range(self.maximum_in_grid):
                    

                        y_pres_p, _, _ = predict_spair(comp_v[:, :, i, j, t])
                        
                        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                            y_pres_batch_single = (y_pres_batch > 0).to(y_pres_batch.dtype)
                            pyro.sample(
                                f"y_pres_{i}_{j}_{t}", 
                                dist.Bernoulli(y_pres_p.cpu()).to_event(1), 
                                obs=y_pres_batch_single)
                            y_pres_batch -= y_pres_batch_single




    # def guide_step(self, n, i, j, images, y_pres, compartment_vector, prev_processed):
    def guide_step_lstm(self, n, i, j, t, images, compartment_vector, prev_z_pres_single_in_compartment):

        device = self.device
        
        encode = self.encode
        predict_spair = self.predict_spair
        baseline_step = self.baseline_step

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width

        n_grid_height_half = n_grid_height / 2.
        n_grid_width_half = n_grid_width / 2.

        # n_grids = n_grid_height * n_grid_width
        n_grids = self.n_grids
        n_counts = n_grids + 1

        bounding_box_scale_min = self.bounding_box_scale_min
        bounding_box_scale_max = self.bounding_box_scale_max

        z_pres_p, z_where_loc, z_where_scale = predict_spair(compartment_vector)

        # Here we compute the baseline value, and pass it to sample.
        baseline_value = baseline_step(compartment_vector).cpu()
        
        z_pres = pyro.sample('z_pres_{}_{}_{}'.format(i, j, t),
                             dist.Bernoulli(z_pres_p.cpu() * prev_z_pres_single_in_compartment)
                                 .to_event(1),
                             infer=dict(baseline=dict(baseline_value=baseline_value.squeeze(-1)))).cuda(device=device)

        z_where = pyro.sample('z_where_{}_{}_{}'.format(i, j, t),
                              dist.Normal(z_where_loc.cpu(), z_where_scale.cpu())
                                  .mask(z_pres.cpu())
                                  .to_event(1)).cuda(device=device)
        
        z_where_resize_x = sigmoid(z_where[:, 0:1])*(bounding_box_scale_max - bounding_box_scale_min)+bounding_box_scale_min
        z_where_resize_y_over_x = sigmoid(z_where[:, 1:2]) * (1.25-0.75) + 0.75
        z_where_resize_y = z_where_resize_x * z_where_resize_y_over_x
        z_where_rescaled = torch.cat(
            [
                z_where_resize_x, 
                z_where_resize_x * z_where_resize_y_over_x,
                z_where_resize_x * (tanh(z_where[:,2:3]) * 0.5 / n_grid_width_half - (j-n_grid_width_half+0.5)/n_grid_width_half),
                z_where_resize_y * (tanh(z_where[:,3:4]) * 0.5 / n_grid_height_half - (i-n_grid_height_half+0.5)/n_grid_height_half)
            ], 
            1
        )
        
        z_where_in_compartment = z_where_rescaled

        
        x_attend = image_to_object(z_where_in_compartment, images)

        z_what_loc, z_what_scale = encode(x_attend)

        z_what = pyro.sample('z_what_{}_{}_{}'.format(i, j, t),
                             dist.Normal(z_what_loc.cpu(), z_what_scale.cpu())
                                 .mask(z_pres.cpu())
                                 .to_event(1)).cuda(device=device)
        

        # return z_pres, z_where_in_compartment, image_to_selected(z_where_in_compartment, images)*z_pres.view(-1, 1, 1, 1) + prev_processed
        return z_pres.cpu()

    def guide_step(self, n, i, j, images, compartment_vector):

        prev_z_pres = 1
        for t in range(self.maximum_in_grid):
            # prev_z_pres, z_where_in_compartment, prev_processed = self.guide_step_lstm(
            #     n, i, j, t,
            #     images, compartment_vector[..., l], 
            #     prev_z_pres, prev_processed
            # )

            prev_z_pres = self.guide_step_lstm(
                n, i, j, t,
                images, compartment_vector[..., t], 
                prev_z_pres
            )

        # return prev_z_pres, z_where_in_compartment, image_to_selected(z_where_in_compartment, images)*z_pres.view(-1, 1, 1, 1) + prev_processed




    def guide(self, dataset):
        # Register networks for optimization.

        # images, y_pres = dataset
        images = dataset["images"]
        y_pres = dataset["y_pres"]

        device = self.device

        compartment = self.compartment
        encode = self.encode
        predict_spair = self.predict_spair
        baseline_step = self.baseline_step

        img_height = self.img_height
        img_width = self.img_width

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width

        guide_step = self.guide_step

        subsample_size = self.subsample_size


        pyro.module("compartment", compartment)
        pyro.module("predict_spair", predict_spair)
        pyro.module("encode", encode)
        pyro.module("baseline_step", baseline_step)

        with pyro.plate('images', images.size(0), subsample_size=subsample_size) as indices:
            n = indices.size(0)
            
            batch = images[indices].view(-1, 1, img_height, img_width).cuda(device=device)
            compartment_vector = compartment(batch)
            # processed = torch.zeros_like(batch).cuda(device=device)
            
            # print("indices: ", indices)

            # steps = []
            for i in range(n_grid_height):
                for j in range(n_grid_width):
                    # z_pres, z_where, processed = guide_step(n, i, j, batch, y_pres[indices], compartment_vector[:, :, i, j], processed)
                    # z_pres, z_where, processed = guide_step(n, i, j, batch, compartment_vector[:, :, i, j], processed)
                    # processed = guide_step(n, i, j, batch, compartment_vector[:, :, i, j], processed)
                    guide_step(n, i, j, batch, compartment_vector[:, :, i, j])
                    # print(processed)
                    # steps.append((z_pres, z_where, processed))
            # return steps




    def elbo(self, model, guide, dataset, *args, **kargs):

        device = self.device

        guide = self.guide
        model = self.model

        img_height = self.img_height
        img_width = self.img_width

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid

        z_pres_counts_prior = self.z_pres_counts_prior

        n_grid_height_half = n_grid_height / 2.
        n_grid_width_half = n_grid_width / 2.

        # n_grids = n_grid_height * n_grid_width
        n_grids = self.n_grids
        n_counts = n_grids + 1      

   
        guide_trace = poutine.trace(guide).get_trace(dataset, *args, **kargs)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(dataset, *args, **kargs)
        
        n = model_trace.nodes["images"]["value"].size(0)
            
        z_pres_p = z_pres_p_init  = z_pres_counts_prior.expand(n, -1)
        
        prev_z_pres = torch.zeros([n, 1])
        loss = 0
        background_left = torch.ones(n, img_height, img_width).cuda(device=device)
        overlap_dist = dist.Gamma(1., 120.)
        for name, site in model_trace.nodes.items():
            
            
            counts = torch.arange(0, n_counts).view(1, n_counts).expand(n, n_counts)
            if site["type"] == "sample":
                if name.startswith("z_pres"):
                    _, _, i, j, t = name.split("_")
                    i, j, t = int(i), int(j), int(t)
                    k = (i*n_grid_width + j) * maximum_in_grid + t + 1
                    
                    
                    z_where_site_name = f"z_where_{i}_{j}_{t}"
                    z_where_temp = model_trace.nodes[z_where_site_name]["value"].cuda(device=device)
                    
                    z_pres_temp = site["value"].cuda(device=device)
                    
                    
                    object_visible = image_to_object(z_where_temp, background_left)
                    object_ratio = 1 - torch.mean(object_visible, axis=-1)
                    object_ratio = torch.clamp(object_ratio, 1e-37)
                    object_score = torch.clamp(object_to_image(z_where_temp, object_visible), 0., 1.)
                    background_left = background_left * (1. - object_score * z_pres_temp.view(n, 1, 1))
                    
                    p_overlap = overlap_dist.log_prob(object_ratio) * z_pres_temp.cpu()
                    
                    # print(object_score.shape, z_pres_temp.shape)
                    

                    # p(z_k|z_{k-1})
                    z_pres_p = z_pres_p * torch.clamp( (counts - prev_z_pres.expand(n, n_counts)) / torch.tensor(n_grids - (k - 1)) , min=0, max=1)
                    # when k >= counts, p(z_k|z_{k-1}) = 0
                    z_pres_p_masked = z_pres_p * torch.gt(counts, prev_z_pres)
                    # Σ_c p(z_k | z_{k-1}, c)
                    z_pres_p_masked_sum = z_pres_p_masked.sum(1, keepdim=True) 
                    z_pres_p_masked_sum = z_pres_p_masked_sum * site["value"] + (1 - z_pres_p_masked_sum) * (1 - site["value"])
                    
                    # optimize the posterior p(θ|x)
                    loss += -(torch.log(z_pres_p_masked_sum)[z_pres_p_masked_sum>0]).sum() 
                    loss += -p_overlap.sum()
                    
                    # p(z_k) calculated twice
                    loss -= -1*(site["fn"].log_prob(site["value"]).sum())

                    
                    prev_z_pres = prev_z_pres + site["value"]

        # loss += TraceGraph_ELBO().loss_and_grads(model, guide, images, *args, **kargs)
        
        # loss += Trace_ELBO().differentiable_loss(model, guide, images, y_pres, *args, **kargs)
        
        loss += -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
        
        return loss
    

    def gen_svi_model(self):

        model = self.model
        guide = self.guide
        optimizer = self.optimizer
        elbo = self.elbo
        # elbo = Trace_ELBO()


        svi = SVI(model,
          guide,
          optimizer,
          loss=elbo
         )

        self.svi = svi

    def train(self):
        device = self.device
        svi = self.svi
        img_height = self.img_height
        img_width = self.img_width

        save_model = self.save_svi
        model_root = self.model_root

        model_id = self.model_id

        max_epoch = self.max_epoch

        # # images, y_pres = dataset = None
        # X_np = torch.load("images/dataset/dsb2018/X").numpy()
        # dsb = torch.from_numpy(X_np)
        # images = dsb.view(-1, img_height * img_width)
        # images = images.cuda(device=device)
        # y_pres = torch.load("images/dataset/dsb2018/y_pres")

        # dataset = (images, y_pres)
        # self.generate_dataset()

        losses = []
        # last_loss = 0
        last_loss = self.loss
        # for i in range(0, 2):
        i = self.current_epoch
        while self.enable_train and i < max_epoch:
            self.current_epoch = i
            self.training = True

            loss = svi.step(self.dataset)


            print(self.__class__.__name__, ": ", 'i={}, elbo={:.2f}'.format(i, loss / self.dataset["images"].size(0)))
            losses.append(loss)
            self.losses = losses

            if loss < last_loss:
                last_loss = loss
                # save_model(f"{model_root}_best/{i}-{loss / images.size(0)}")
                self.loss = loss
                save_model()
                # save_model(suffix=True)

            if i%1000 == 0:
                model_root_sub =os.path.join(model_home, self.__class__.__name__, model_id)
                os.makedirs(model_root_sub, exist_ok=True)
                shutil.copytree(model_root, model_root_sub, dirs_exist_ok=True)
            

            i += 1
            self.current_epoch = i % max_epoch

        self.training = False

    def predict(self, images):
        # model = self.model
        self.load_svi()

        guide = self.guide
        prior = self.prior

        compartment = self.compartment
        predict_spair = self.predict_spair

        subsample_size = self.subsample_size
        img_height = self.img_height
        img_width = self.img_width
        bounding_box_scale_max = self.bounding_box_scale_max
        bounding_box_scale_min = self.bounding_box_scale_min

        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid

        n_grid_height_half = n_grid_height / 2.
        n_grid_width_half = n_grid_width / 2.

        n_images = images.shape[0]

        print("images.max: ", images.max())
        print("images.min: ", images.min())
        print("images.mean: ", images.mean())

        if self.dataset_min is None or self.dataset_max is None:
            self.dataset_min = dataset_min = images.min()
            self.dataset_max = dataset_max = images.max()
        else:
            dataset_min = self.dataset_min
            dataset_max = self.dataset_max

        images = (images - dataset_min) / max(1, dataset_max- dataset_min)


        n_batchs = np.ceil(n_images / subsample_size)

        # x = []
        z_where = []
        z_pres = []

        for i in range(0, n_images, subsample_size):
            imgs = images[i:i+subsample_size]
            n_subset = imgs.shape[0]
            if n_subset < subsample_size:
                temp = torch.zeros([subsample_size]+list(images.shape[1:]), dtype=images.dtype)
                temp[:n_subset] = imgs
                imgs = temp

            # subset_ = [subset,]
            # for i in range(subsample_size - n_subset):
            #     subset_.append(subset[0:1])
            # subset = torch.cat(subset_, axis=0)    

            # subset = imgs, [None] * subsample_size

            # trace = poutine.trace(guide).get_trace(subset)
            # x_, z_where_, z_pres_ = poutine.replay(prior, trace=trace)(subsample_size)

            z_where_ = []
            z_pres_ = []
            compartment_vector = compartment(imgs.cuda().view(-1, 1, img_height, img_width))
            for gi in range(n_grid_height):
                for gj in range(n_grid_width):
                    for gt in range(maximum_in_grid):
                        z_pres_p, z_where_loc, z_where_scale = predict_spair(compartment_vector[..., gi, gj, gt])

                        z_where_resize_x = sigmoid(z_where_loc[:, 0:1])*(bounding_box_scale_max - bounding_box_scale_min)+bounding_box_scale_min
                        z_where_resize_y_over_x = sigmoid(z_where_loc[:, 1:2]) * (1.25-0.75) + 0.75
                        z_where_resize_y = z_where_resize_x * z_where_resize_y_over_x
                        z_where_rescaled = torch.cat(
                            [
                                z_where_resize_x, 
                                z_where_resize_x * z_where_resize_y_over_x,
                                z_where_resize_x * (tanh(z_where_loc[:,2:3]) * 0.5 / n_grid_width_half - (gj-n_grid_width_half+0.5)/n_grid_width_half),
                                z_where_resize_y * (tanh(z_where_loc[:,3:4]) * 0.5 / n_grid_height_half - (gi-n_grid_height_half+0.5)/n_grid_height_half)
                            ], 
                            1
                        )
                        

                        z_where_.append(z_where_rescaled)
                        z_pres_.append(z_pres_p>0.5)




            z_where_ = torch.stack(z_where_, axis=0)
            z_pres_ = torch.stack(z_pres_, axis=0)

            z_where_ = torch.swapaxes(z_where_, 0, 1)
            z_pres_ = torch.swapaxes(z_pres_, 0, 1)



            # x.append(x_[:n_subset])
            z_where.append(z_where_[:n_subset])
            z_pres.append(z_pres_[:n_subset])

        # x = torch.cat(x, axis=0)
        z_where = torch.cat(z_where, axis=0)
        z_pres = torch.cat(z_pres, axis=0)

        print("z_where.shape: ", z_where.shape)
        print("z_pres.shape: ", z_pres.shape)

        return z_where.cpu().detach().numpy(), \
            z_where_inv(z_where.view(-1, 4)).view(z_where.shape).cpu().detach().numpy(), z_pres.cpu().detach().numpy()



    def load_svi(self, reload=False):
        save_table = self.save_table

        print("Loading model...")

        if self.loaded == True and reload == False:
            print("Model is already loaded.")
            return False


        model_root = self.model_root

        encode = self.encode
        decode = self.decode
        compartment = self.compartment
        predict_spair = self.predict_spair
        baseline_step = self.baseline_step
        optimizer = self.optimizer
        param_store = self.param_store = pyro.get_param_store()

        encode_path = self.encode_path
        decode_path = self.decode_path
        compartment_path = self.compartment_path
        predict_spair_path = self.predict_spair_path
        baseline_step_path = self.baseline_step_path
        optim_path = self.optim_path
        model_path = self.model_path
        config_path = self.config_path


        # if "encode" in save_table and os.path.exists(encode_path):
        if os.path.exists(encode_path):
            try:
                encode.load_state_dict(torch.load(encode_path))
            except Exception as e:
                print(e, "\n")
        else:
            print("No encode exist. ")


        # if "decode" in save_table and os.path.exists(decode_path):
        if os.path.exists(decode_path):

            try:
                decode.load_state_dict(torch.load(decode_path))
            except Exception as e:
                print(e, "\n")
        else:
            print("No decode exist. ")

        # if "compartment" in save_table and os.path.exists(compartment_path):
        if os.path.exists(compartment_path):
            try:
                compartment.load_state_dict(torch.load(compartment_path))
            except Exception as e:
                print(e, "\n")
        else:
            print("No compartment exist. ")

        # if "predict_spair" in save_table and os.path.exists(predict_spair_path):
        if os.path.exists(predict_spair_path):
            try:
                predict_spair.load_state_dict(torch.load(predict_spair_path))
            except Exception as e:
                print(e, "\n")
        else:
            print("No predict_spair exist. ")

        # if "baseline_step" in save_table and os.path.exists(baseline_step_path):
        if os.path.exists(baseline_step_path):
            try:
                baseline_step.load_state_dict(torch.load(baseline_step_path))
            except Exception as e:
                print(e, "\n")
        else:
            print("No baseline_step exist. ")

        # if "optimizer" in save_table and os.path.exists(optim_path):
        if os.path.exists(optim_path):
            try:
                optimizer.load(optim_path)
            except Exception as e:
                print(e, "\n")
        else:
            print("No optimizer exist. ")


        # if "model" in save_table and os.path.exists(model_path):
        if os.path.exists(model_path):
            try:
                param_store.load(model_path)
            except Exception as e:
                print(e, "\n")
        else:
            print("No model exist. ")

        # if "config" in save_table and os.path.exists(config_path):
        if os.path.exists(config_path):
            try:
                with open(config_path) as fd:
                    config = json.load(fd)
                self.current_epoch = config.get("current_epoch", 0)
                self.current_loss = config.get("current_loss", float("inf")) 
                print("current_epoch: ", self.current_epoch)
            except Exception as e:
                print(e)
                config = {}

        else:

            print("No config exist. ")
            # return False

        self.loaded = True


    # def save_svi(self, suffix=False):
    def save_svi(self):
        # save_table = ["encode", "decode", "compartment", "predict_spair", "baseline", "model", "optimizer", "config"]
        save_table = self.save_table

        print("Saving model...")

        
        model_root = self.model_root
        # if suffix == True:
        # model_root = os.path.join(model_root, self.__class__.__name__)
        os.makedirs(model_root, exist_ok=True)

        encode = self.encode
        decode = self.decode
        compartment = self.compartment
        predict_spair = self.predict_spair
        baseline_step = self.baseline_step
        optimizer = self.optimizer
        param_store = self.param_store = pyro.get_param_store()


        encode_path = self.encode_path
        decode_path = self.decode_path
        compartment_path = self.compartment_path
        predict_spair_path = self.predict_spair_path
        baseline_step_path = self.baseline_step_path
        optim_path = self.optim_path
        model_path = self.model_path
        config_path = self.config_path
        
        if "encode" in save_table:
            torch.save(encode.state_dict(), encode_path)
        if "decode" in save_table:
            torch.save(decode.state_dict(), decode_path)
        if "compartment" in save_table:
            torch.save(compartment.state_dict(), compartment_path)
        if "predict_spair" in save_table:
            torch.save(predict_spair.state_dict(), predict_spair_path)
        if "baseline_step" in save_table:
            torch.save(baseline_step.state_dict(), baseline_step_path)
        if "optimizer" in save_table:
            optimizer.save(optim_path)
        if "model" in save_table:
            param_store.save(model_path)
        if "config" in save_table:
            with open(config_path, mode="w") as fd:
                fd.write(
                    json.dumps({
                        "current_epoch": self.current_epoch,
                        "current_loss": self.loss
                    })
                )

        print("Model saved.")


    def reset_svi(self):

        print("Resetting model...")

        model_root = self.model_root

        try:
            shutil.rmtree(f"{model_root}")
        except Exception as e:
            print(e)


        self.init_components()
        
        

        

        print("Model reset.")

    def generate_dataset(self):
        model_id = self.model_id
        img_height = self.img_height
        img_width = self.img_width
        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width
        maximum_in_grid = self.maximum_in_grid

        sql_engine = create_engine(mysqlurl, pool_recycle=3600)
        sql_inspector = inspect(sql_engine)

        # img_height, img_width, _, _, n_grid_height, n_grid_width, maximum_in_grid = get_model_info(model_id)

        
        sql = f"\
            SELECT \
                t1.cropped_id, t1.path, \
                t3.object_min, t3.object_max, \
                t4.annotation as algorithm_annotation, \
                t4.y_pres as algorithm_y_pres, \
                t5.annotation as manual_annotaion, \
                t5.y_pres as manual_y_pres \
            FROM \
                 cellist.cellist_cropped t1  LEFT JOIN cellist.cellist_models t3 \
                        ON t1.model_id=t3.model_id \
                            LEFT JOIN cellist_annotation_algorithm t4 \
                                ON t1.cropped_id=t4.cropped_id\
                                    LEFT JOIN cellist_annotation_manual t5\
                                        ON t1.cropped_id = t5.cropped_id \
            WHERE t1.model_id='{model_id}'\
            "


        query = pd.read_sql(
            sql,
            con=sql_engine
        )
        query = query.values

        images = np.zeros((query.shape[0], img_height, img_width))
        # y_pres_algorithm = np.ones((query.shape[0], n_grid_height, n_grid_width)) * -1
        # y_pres_manual = np.ones((query.shape[0], n_grid_height, n_grid_width)) * -1
        y_pres = np.ones((query.shape[0], n_grid_height, n_grid_width)) * -1


        index_mapping = BiDict()
        for i, (
                cropped_id, path, 
                object_min, object_max, 
                algorithm_annotation, algorithm_y_pres, 
                manual_annotaion, manual_y_pres
            ) in enumerate(query):

            # object_size = np.log(object_min) + np.log(object_max)
            # object_size /= 2
            # object_size = np.exp(object_size)

            img = io.imread(path)

            if len(img.shape) == 3:
                if img.shape[-1] == 1:
                    gray = img[..., 0]
                elif img.shape[-1] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[-1] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    print("img.shape: ", img.shape)
                    raise(Exception("Why such a weird shape? "))
            else:
                gray = img

            images[i] = gray



            # if algorithm_y_pres is not None:
            #     y_pres_algorithm[i] = algorithm_y_pres

            if algorithm_y_pres is not None:
                y_pres[i] = json.loads(algorithm_y_pres)


            if manual_y_pres is not None:
                y_pres[i] = json.loads(manual_y_pres)

            index_mapping[i] = cropped_id


        self.dataset_min = dataset_min = images.min()
        self.dataset_max = dataset_max = images.max()
        print("images.max: ", dataset_min)
        print("images.min: ", dataset_max)
        print("images.mean: ", images.mean())
        images = (images - dataset_min) / max(dataset_max - dataset_min, 1)

        self.dataset = OrderedDict(
            (
                ("images", torch.from_numpy(images).to(torch.float32).view(-1, img_height*img_width)), 
                # ("y_pres_algorithm", torch.from_numpy(y_pres_algorithm)), 
                # ("y_pres_manual", torch.from_numpy(y_pres_manual)), 
                # ("y_pres", torch.from_numpy(y_pres>0).to(torch.float32)),
                ("y_pres", torch.from_numpy(y_pres).to(torch.float32)),
                ("mapping", index_mapping)
            )
        )


    def update_dataset(self, cropped_id, annotation):
        
        self.dataset["y_pres"][self.dataset["mapping"].inverse[cropped_id]] = annotation




if __name__ == "__main__":

    ModelD2Init("hhah")

    pass




