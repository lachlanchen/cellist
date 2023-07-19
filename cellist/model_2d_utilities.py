import os
from collections import namedtuple, OrderedDict
import numpy as np

from sqlalchemy import inspect, create_engine
import pandas as pd
import pymysql as pm

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



# from cellist.model_2d_constants import img_height, img_width, obj_height, obj_width
from cellist.utils.constants import curdir, dataroot, mysqlconfig, mysqlurl


class BiDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = OrderedDict()
        for key, value in self.items():
            # self.inverse.setdefault(value, []).append(key) 
            self.inverse.setdefault(value, key)

    def __setitem__(self, key, value):
        if key in self:
            # self.inverse[self[key]].remove(key) 
            del self.inverse[self[key]]
        super(BiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, key)

    def __delitem__(self, key):
        # self.inverse.setdefault(self[key], []).remove(key)
        del self.inverse[self[key]]
        # if self[key] in self.inverse and not self.inverse[self[key]]: 
        #     del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)

def get_model_info(model_id):
    conn = pm.connect(
        **mysqlconfig
    )
    conn.select_db("cellist")
    cursor = conn.cursor()
    sql = f" \
        SELECT \
            slice_height, slice_width, object_min, object_max, n_grid_height, n_grid_width, maximum_in_grid\
        FROM \
            cellist_models \
        WHERE \
            model_id = \'{model_id}\'\
        LIMIT \
            1 \
        "
    status = cursor.execute(sql)
    model_info = cursor.fetchone()

    slice_height, slice_width, object_min, object_max, n_grid_height, n_grid_width, maximum_in_grid = model_info

    return slice_height, slice_width, object_min, object_max, n_grid_height, n_grid_width, maximum_in_grid


def get_grid_info(model_id):
    # conn = pm.connect(
    #     **mysqlconfig
    # )
    # conn.select_db("cellist")
    # cursor = conn.cursor()
    # sql = f" \
    #     SELECT \
    #         n_grid_height, n_grid_width, maximum_in_grid\
    #     FROM \
    #         cellist_models \
    #     WHERE \
    #         model_id = \'{model_id}\'\
    #     LIMIT \
    #         1 \
    #     "
    # status = cursor.execute(sql)
    # grid_info = cursor.fetchone()

    # n_grid_height, n_grid_width, maximum_in_grid = grid_info

    _, _, _, _, n_grid_height, n_grid_width, maximum_in_grid = get_model_info(model_id)

    return n_grid_height, n_grid_width, maximum_in_grid


def expand_z_where(z_where, device=None):
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [sx, sy, x, y] -> [[sx,0,x],
    #                    [0,sy,y]]
    n = z_where.size(0)
    expansion_indices = torch.cuda.LongTensor([1, 0, 3, 0, 2, 4], device=device)
    out = torch.cat((torch.zeros([1, 1]).expand(n, 1).cuda(device=device), z_where), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

def object_to_image(z_where, obj, obj_height=100, obj_width=100, img_height=256, img_width=256, device=None):
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

def z_where_inv(z_where, device=None):
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

def image_to_object(z_where, image, obj_height=100, obj_width=100, img_height=256, img_width=256, device=None):
    n = image.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv, torch.Size((n, 1, obj_height, obj_width)))
    out = grid_sample(image.view(n, 1, img_height, img_width), grid)
    return out.view(n, -1)

def image_to_selected(z_where, image, obj_height=100, obj_width=100, img_height=256, img_width=256, device=None):
    n = image.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv, torch.Size((n, 1, obj_height, obj_width)))
    out = grid_sample(image.view(n, 1, img_height, img_width), grid)
    
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n ,1, img_height, img_width)))
    out = grid_sample(out.view(n, 1, obj_height, obj_width), grid)
    
    # return out.view(n, -1)
    return out





def add_square(z_where, obj):
    # print("z_where.shape: ", z_where.shape)
    # print("obj.shape: ", obj.shape)
    
    n = obj.size(0)
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n, 1,img_height, img_width)))
    out = grid_sample(obj.view(n, 1, obj_height, obj_width), grid)
    return out.view(n, img_height, img_width)

def add_square_all(images, pres, where):
    
    obj = torch.zeros((images.shape[0], 100, 100), device=device)

    obj[:, :5] = 0.2
    obj[:, -5:] = 0.2

    obj[:, :, :5] = 0.2
    obj[:, :, -5:] = 0.2
    
    prev_pres = torch.zeros_like(pres[0], device=device)
    
    for i in range(64):
        images += add_square(where[i], obj) * (pres[i] > prev_pres)[..., None]
        prev_pres = pres[i]
        
    return images


# pres = torch.stack(z_pres, axis=0)
# where = torch.stack(z_where, axis=0)

# images = add_square_all(torch.from_numpy(X_np[0+start_idx:4+start_idx]).cuda(device=device), pres.cuda(device=device), where)

# images.shape

# show_images(images.cpu())