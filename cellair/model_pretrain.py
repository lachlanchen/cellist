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



from cellair.dicton import Dicton
from cellair.model_init import ModelD2Init
from cellair.model_2d_components import Decoder, Encoder, Compartment, PredictSPAIR, BaselineStep
from cellair.model_2d_utilities import *
# from cellair.model_2d_constants import *
from cellair.utils.constants import curdir, dataroot, mysqlconfig, mysqlurl

import json
import shutil

# smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.0')

model_home = "data/models"

torch.set_default_dtype(torch.float32)

class ModelD2Pretrain(ModelD2Init):


    def __del__(self):
        pass

    def __init__(self, model_id, lr=1e-4, *args, **kargs):
        self.init_constants(model_id, lr=1e-5, *args, **kargs)

        self.save_table = ["encode", "decode", "config"]
        self.max_epoch = 1000000
        self.subsample_size = 128
        # self.lr = 1e-3

        self.generate_dataset()
        self.init_components()


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

        self.decode=Decoder(
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

        # self.decode.train(False)
        # self.encode.train(False)
    

        self.gen_svi_model()

        self.load_svi()
        self.save_svi()

        print("Assembled.")


    def prior_step(
        self, 
        n):

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

        z_what = pyro.sample('z_what',
                             dist.Normal(z_what_prior_loc.expand(n, -1),
                                         z_what_prior_scale.expand(n, -1))
                                 .to_event(1)).to(device=device)

        y_attend_loc, y_attend_scale = decode(z_what)
        
        return y_attend_loc, y_attend_scale



    


    def model(self, dataset):
        device = self.device

        decode = self.decode
        compartment = self.compartment
        predict_spair = self.predict_spair

        prior = self.prior_step

        img_height = self.img_height
        img_width = self.img_width
        obj_height = self.obj_height
        obj_width = self.obj_width
        n_grid_height = self.n_grid_height
        n_grid_width = self.n_grid_width

        # images, y_pres = dataset
        # images = dataset["images"]
        # y_where = dataset["y_where"]
        objects = dataset["objects"]

        # Register network for optimization.
        pyro.module("decode", decode)
        # pyro.module("compartment", compartment)
        # pyro.module("predict_spair", predict_spair)
        with pyro.plate('objects', objects.size(0)) as indices:
            n = indices.size(0)

            batch = objects[indices].to(device=device)

           

            # batch = batch.to(device=device)

            objects_loc, objects_scale = prior(n)

            # print("batch.max: ", batch.max())
            # print("objects.max: ", objects_loc.max())


            # sd = (0.3 * torch.ones(1)).expand_as(objects)
            pyro.sample(
                'obs', 
                dist.Normal(objects_loc.cpu(), objects_scale.cpu()).to_event(1),
                obs=batch.cpu()
            )






    def guide_step(self, n, objects):

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


        

        z_what_loc, z_what_scale = encode(objects)

        z_what = pyro.sample('z_what',
                             dist.Normal(z_what_loc.cpu(), z_what_scale.cpu())
                                 .to_event(1)
                                 ).to(device=device)
        




    def guide(self, dataset):
        # Register networks for optimization.

        # images, y_pres = dataset
        # images = dataset["images"]
        # y_pres = dataset["y_pres"]
        # y_where = dataset["y_where"]
        objects = dataset["objects"]

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


        # pyro.module("compartment", compartment)
        # pyro.module("predict_spair", predict_spair)
        pyro.module("encode", encode)
        # pyro.module("baseline_step", baseline_step)

        with pyro.plate('objects', objects.size(0), subsample_size=subsample_size) as indices:
            n = indices.size(0)
            batch = objects[indices]

            guide_step(n, batch)


    def gen_svi_model(self):

        model = self.model
        guide = self.guide
        optimizer = self.optimizer
        # elbo = self.elbo
        elbo = Trace_ELBO()


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
        last_loss = self.loss
        # for i in range(0, 2):
        i = self.current_epoch
        while self.enable_train and i < max_epoch:
            self.current_epoch = i
            self.training = True

            loss = svi.step(self.dataset) / self.dataset["objects"].size(0)


            print(self.__class__.__name__, ": ", 'i={}, elbo={:.2f}'.format(i, loss))
            losses.append(loss)
            self.losses = losses


            # print(loss, last_loss)
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
                self.predict()
            

            i += 1
            self.current_epoch = i % max_epoch

        self.training = False


    def predict(self):
        objects = self.dataset["objects"]
        device = self.device
        obj_height = self.obj_height
        obj_width = self.obj_width

        batch = objects[:10].to(device=device).view(10, -1)

        print("batch.max: ", batch.max)

        latent, _ = self.encode(batch.detach())
        batch_generated, _ = self.decode(latent.detach())

        os.makedirs("objects_decoded", exist_ok=True)
        for i, b in enumerate(batch.cpu()):
            io.imsave(f"objects_decoded/{i}-original.png", b.view(obj_height, obj_width).numpy())
            # io.imsave(f"objects_decoded/{i}-original.png", b)
        for i, o in enumerate(batch_generated.cpu()):
            io.imsave(f"objects_decoded/{i}-generated.png", o.detach().view(obj_height, obj_width).numpy())




    def generate_dataset(self):
        device = self.device

        model_id = self.model_id
        img_height = self.img_height
        img_width = self.img_width
        obj_height = self.obj_height
        obj_width = self.obj_width
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
                t5.annotation as manual_annotation, \
                t5.y_pres as manual_y_pres \
            FROM \
                 cellair.cellair_cropped t1  LEFT JOIN cellair.cellair_models t3 \
                        ON t1.model_id=t3.model_id \
                            LEFT JOIN cellair_annotation_algorithm t4 \
                                ON t1.cropped_id=t4.cropped_id\
                                    LEFT JOIN cellair_annotation_manual t5\
                                        ON t1.cropped_id = t5.cropped_id \
            WHERE t1.model_id='{model_id}'\
            "


        query = pd.read_sql(
            sql,
            con=sql_engine
        )
        query = query.values

        images = np.zeros((query.shape[0], img_height, img_width), dtype=np.float32)
        objects = []
        # y_where = []




        index_mapping = BiDict()
        for i, (
                cropped_id, path, 
                object_min, object_max, 
                algorithm_annotation, algorithm_y_pres, 
                manual_annotation, manual_y_pres
            ) in enumerate(query):

            # object_size = np.log(object_min) + np.log(object_max)
            # object_size /= 2
            # object_size = np.exp(object_size)

            img = io.imread(path)

            if len(img.shape) == 3:
                if img.shape[-1] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[-1] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:
                gray = img

            gray = gray.astype(np.float32)
            images[i] = gray
            # images.append(gray)
            gray = torch.from_numpy(gray)
            gray = gray.view(-1, 1, img_height, img_width).to(device=device)




            # print(algorithm_annotation)
            # print(manual_annotation)

            if manual_annotation is not None:
                annotation = json.loads(manual_annotation)
                
            elif algorithm_annotation is not None:
                annotation = json.loads(algorithm_annotation)
            else:
                continue

            annotation = torch.tensor(annotation).to(device=device)
            annotation = z_where_inv(annotation)

            # print(annotation.shape, gray.shape)

            # print(annotation)
            # print(gray)
            objects_single = image_to_object(annotation, gray.repeat(annotation.size(0), 1, 1, 1))

            if i == 0:
                os.makedirs("objects", exist_ok=True)
                for j, obj in enumerate(objects_single.reshape(-1, obj_height, obj_width).cpu()):
                    print(obj.shape)
                    io.imsave(f"objects/{j}.png", obj)


            objects.append(objects_single)



            index_mapping[i] = cropped_id


        images = np.stack(images, axis=0)

        objects = torch.cat(objects, axis=0)
        print("objects.shape: ", objects.shape)


        self.dataset_min = dataset_min = images.min()
        self.dataset_max = dataset_max = images.max()
        print("images.max: ", dataset_min)
        print("images.min: ", dataset_max)
        print("images.mean: ", images.mean())
        images = (images - dataset_min) / max(dataset_max - dataset_min, 1)
        objects = (objects - dataset_min) / max(dataset_max - dataset_min, 1)

        self.dataset = OrderedDict(
            (
                ("images", torch.from_numpy(images).to(torch.float32).view(-1, img_height*img_width)), 
                ("objects", objects),
                ("mapping", index_mapping)
            )
        )


    def update_dataset(self, cropped_id, annotation):
        
        # self.dataset["y_pres"][self.dataset["mapping"].inverse[cropped_id]] = annotation
        pass




if __name__ == "__main__":

    ModelD2Pretrain("hhah")

    pass




