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

from model import z_where_inv, guide, prior

import json




smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.0')


device = torch.device('cuda:0')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
	X_np = torch.load("X").numpy()
	dsb = torch.from_numpy(X_np)
	y_pres = torch.load("y_pres")

	N, img_height, img_width = X_np.shape


	data = dsb.view(-1, img_height * img_width)

	start_idx = 30
	trace = poutine.trace(guide).get_trace(data[start_idx+0:start_idx+4], y_pres[start_idx+0:start_idx+4])
	x, z_where, z_pres = poutine.replay(prior, trace=trace)(data[start_idx+0:start_idx+4].size(0))

	z_where = torch.stack(z_where, axis=0).view(4*64, -1)
	z_where_inverse = z_where_inv(z_where).view(64, 4, -1)

	print(z_where)
	print(z_where_inverse)
	z_where = z_where.cpu().detach().numpy()
	z_where_inverse = z_where_inverse.cpu().detach().numpy()
	z_pres = torch.stack(z_pres, axis=0).cpu().detach().numpy()
	print(z_pres.shape)

	z_where_inverse = np.swapaxes(z_where_inverse, 0, 1)
	z_pres = np.swapaxes(z_pres, 0, 1)
	z_pres = np.diff(z_pres, axis=1, prepend=z_pres[:, 0:1])

	z_where_inverse_result = {}
	for i in range(4):
		z_where_inverse_result[i] = z_where_inverse[i]
	with open("z_where.json", mode="w") as fd:
		fd.write(json.dumps(z_where_inverse_result, cls=NumpyEncoder))

	z_pres_result = {}
	for i in range(4):
		z_pres_result[i] = z_pres[i, ..., 0]
	with open("z_pres.json", mode="w") as fd:
		fd.write(json.dumps(z_pres_result, cls=NumpyEncoder))


	from skimage import io
	for i in range(4):
		io.imsave(f"{i}.png", data[start_idx+i].view(img_height, img_width).cpu().detach().numpy())


	print(z_where)