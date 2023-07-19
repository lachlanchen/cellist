from skimage import io

import os
import json
import re

import tornado.ioloop
import tornado.web
import tornado.websocket

from tornado import gen
from tornado.options import define, options, parse_command_line

from concurrent.futures import ThreadPoolExecutor


import time
from datetime import datetime
import random
import string

import numpy as np
import pandas as pd
import pymysql as pm
from sqlalchemy import inspect, create_engine

from cellist.dicton import Dicton
from cellist.model_init import ModelD2Init
from cellist.model_pretrain import ModelD2Pretrain
from cellist.utils.constants import curdir, dataroot, mysqlconfig, mysqlurl
from cellist.image_preprocessing import image_slice, image_stitch
from cellist.model_2d_utilities import get_grid_info

from uuid import uuid4

import torch

import gc
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

from cellpose import models

from skimage import io

uuid_cropped = lambda x: str(x)+"-"+str(uuid4())

def worker_callbacks(f):
    e = f.exception()

    if e is None:
        # return f.result()
        # return f
        return None

    trace = []
    tb = e.__traceback__
    while tb is not None:
        trace.append({
            "filename": tb.tb_frame.f_code.co_filename,
            "name": tb.tb_frame.f_code.co_name,
            "lineno": tb.tb_lineno
        })
        tb = tb.tb_next
    print(json.dumps({
        'type': type(e).__name__,
        'message': str(e),
        'trace': trace
    }, indent=4).replace('\\n', '\n').replace('\\t', '\t'))


thread_pool = ThreadPoolExecutor(max_workers=64)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rank_check(img):

    shape = img.shape

    rank_num = 2
    n_channels = 1
    height = None
    width = None
    depth = None

    if len(shape) == 2:
        rank_num = 2
        n_channels = 1
        height, width = shape

        gray = img
    elif len(shape) == 3:

        if shape[-1] <= 4:
            rank_num = 2
            height, width, n_channels = shape

            gray = img[..., 0]

        else:
            rank_num = 3
            n_channels = 1
            depth, height, width = shape

            gray = img

    else:
        raise Exception("Unrecognized image format.")

    return gray, rank_num, n_channels, height, width, depth

def create_model_dataset(model_id):
    pass



def image_slice_2d(image, model_height=256, model_width=256, n_channels=3):

    model_height_half = model_height // 2
    model_width_half = model_width // 2

    images = []
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, _ = cv2.threshold(gray.astype(np.uint8), 0, color_depth, cv2.THRESH_OTSU)
    ret = np.mean(image)

    height = image.shape[0]
    width = image.shape[1]
    n_window_height = np.ceil(height / model_height).astype(int)
    n_window_width = np.ceil(width / model_width).astype(int)
    height_holder = (n_window_height * model_height).astype(int)
    width_holder = (n_window_width * model_width).astype(int)

    image_holder = np.zeros((height_holder, width_holder, n_channels), dtype=image.dtype) + ret
    image_holder[:height, :width] = image
    elem_len = image_holder.strides[-1]

    # images = np.zeros((n_window_width*n_window_height, 256, 256, 3), dtype=image.dtype)
    # for i in range(n_window_height):
    #     for j in range(n_window_width):
    #        images[i*n_window_width + j] = image_holder[i*256:(i+1)*256, j*256:(j+1)*256]


    images = np.lib.stride_tricks.as_strided(
         image_holder, 
         shape=(n_window_height, n_window_width, model_height, model_width, n_channels), 
         strides=(model_height*n_window_width*model_width*n_channels*elem_len, model_width*n_channels*elem_len, n_window_width*model_width*n_channels*elem_len, n_channels*elem_len, elem_len)
    )
    images = images.reshape(-1, model_height, model_width, n_channels)


    pos_height = np.array([[i*model_height] * n_window_width for i in range(n_window_height)]).reshape(-1)
    pos_width = np.array([i*model_width for i in range(n_window_width)] * n_window_height).reshape(-1)
    uuids = np.array([[uuid_cropped(f"cropped-{i:02}-{j:02}") for j in range(n_window_width)] for i in range(n_window_height)]).reshape(-1)

    # image_holder_overlap = np.zeros((height_holder+model_height, width_holder+model_width, n_channels), dtype=image.dtype) + ret
    # image_holder_overlap[model_height_half:height+model_height_half, model_width_half:width+model_width_half] = image

    # images_overlap = np.lib.stride_tricks.as_strided(image_holder_overlap, 
    #      shape=(n_window_height+1, n_window_width+1, model_height, model_width, n_channels), 
    #      strides=(model_height*(n_window_width+1)*model_width*n_channels*elem_len, model_width*n_channels*elem_len, (n_window_width+1)*model_width*n_channels*elem_len, n_channels*elem_len, elem_len))
    # images_overlap = images_overlap.reshape(-1, model_height, model_width, n_channels)


    # return images, images_overlap, (height, width)
    return images, height, width, pos_height, pos_width, uuids


def crop_images_daemon(model_id, images, model_height=256, model_width=256):
    sql_engine = create_engine(mysqlurl, pool_recycle=3600)
    sql_inspector = inspect(sql_engine)

    images_query_array =  '"' +'","'.join(images) + '"'
    query_ql = f'select image_id, path from cellist_images where image_id in( {images_query_array} ) '
    # print(query_ql)
    query = pd.DataFrame([])
    while len(images) != len(query):
        query = pd.read_sql(
            query_ql, 
            con=sql_engine)
    query = query.values


    data = pd.DataFrame()

    for idx, (image_id, path) in enumerate(query):
        img = io.imread(path)

        filename, img_extension = os.path.splitext(path)

        gray, rank_num, n_channels, height, width, depth = rank_check(img)

        # print("img.shape: ", img.shape)

        if rank_num == 2:
            images_cropped, height, width, pos_height, pos_width, ids = image_slice_2d(img.reshape(height, width, n_channels), model_height=model_height, model_width=model_width, n_channels=n_channels)

            # images_cropped = np.squeeze(images_cropped)
            

            # print(pos_height)
            # print(pos_width)
            # print(ids)

            root = "data/cropped"
            root_image = os.path.join(root, model_id, image_id)
            os.makedirs(root_image, exist_ok=True)
            paths_small = [os.path.join(root_image, small_id+img_extension) for small_id in ids]
            for img_idx, img_small in enumerate(images_cropped):
                io.imsave(paths_small[img_idx], img_small)


            data_single = pd.DataFrame(
                {
                    "cropped_id": ids,
                    "path": paths_small,
                    "pos_y": pos_height,
                    "pos_x": pos_width,

                }
            )

            data_single["model_id"] = model_id

            data_single["rank_num"] = rank_num
            data_single["n_channels"] = n_channels

            data_single["image_id"] = image_id
            data_single["height"] = model_height
            data_single["width"] = model_width

            data = data.append(data_single)

        else:
            print("This is a 3D image. ")



    table = "cellist_cropped"
    check = sql_inspector.has_table(table)
    if check:
        data.to_sql(table, con=sql_engine, index=False, if_exists="append")
    else:
        data.to_sql(table, con=sql_engine, index=False, if_exists="replace")


    return True, "Success"



    # print(query)

    # while True:
    #     print("haha")
    #     time.sleep(3)





def update_model_images(model_id, images):

    sql_engine = create_engine(mysqlurl, pool_recycle=3600)
    sql_inspector = inspect(sql_engine)

    data = pd.DataFrame({"image_id": images}, index=list(range(len(images))))
    data["model_id"] = model_id



    table = "cellist_model_images_rel"
    check = sql_inspector.has_table(table)
    if check:
        data.to_sql(table, con=sql_engine, index=False, if_exists="append")
    else:
        data.to_sql(table, con=sql_engine, index=False, if_exists="replace")

    return True, model_id, "Success"





def create_model_info(arguments):

    sql_engine = create_engine(mysqlurl, pool_recycle=3600)
    sql_inspector = inspect(sql_engine)
    

    now = datetime.now()
    submission_time = now
    now_string = now.strftime("%Y%m%d_%H%M%S_%f")
    model_id = f'{arguments["username"]}_{arguments["model_name"]}_{now_string}'


    arguments_stringified = {}

    for key in arguments:
        if type(arguments[key]) is list:
            arguments_stringified[key] = ",".join(arguments[key])
        else:
            arguments_stringified[key] = arguments[key]

    arguments_stringified.pop("load_model_id")
    arguments_stringified.pop("password")
    arguments_stringified.pop("local_images")
    arguments_stringified.pop("annotation")
    arguments_stringified.pop("cursor")
    arguments_stringified.pop("last_valid_cursor")
    arguments_stringified.pop("target_image_id")
    arguments_stringified.pop("initialize_algorithm")
    # arguments.pop("local_images_uuid")

    print("arguments_stringified: ", arguments_stringified)

    data = pd.DataFrame(arguments_stringified, index=list(range(1)))
    data["model_id"] = model_id


    table = "cellist_models"
    check = sql_inspector.has_table(table)
    if check:
        data.to_sql(table, con=sql_engine, index=False, if_exists="append")
    else:
        data.to_sql(table, con=sql_engine, index=False, if_exists="replace")


    return True, model_id, "Success"


def check_annotation(model_id, cropped_id, table="algorithm"):
    conn = pm.connect(
        **mysqlconfig
    )
    conn.select_db("cellist")
    cursor = conn.cursor()
    sql = f" \
        SELECT \
            id \
        FROM \
             cellist_annotation_{table}\
        WHERE \
            model_id = '{model_id}' and cropped_id = '{cropped_id}' \
        LIMIT \
            1 \
        "
    status = cursor.execute(sql)
    annotation_id = cursor.fetchone()

    if annotation_id is not None:

        return annotation_id[0]
    else:
        return -1


def update_annotation(annotation_id, annotation, y_pres, table="algorithm"):
    conn = pm.connect(
        **mysqlconfig
    )
    conn.select_db("cellist")
    cursor = conn.cursor()
    sql = f" \
        UPDATE cellist_annotation_{table}\
        SET \
            annotation='{annotation}', \
            y_pres='{y_pres}' \
        WHERE \
            id = '{annotation_id}'\
        "

    cursor.execute(sql)
    result = conn.commit()

    return True, annotation_id, "Success"


def save_manual_annotation(username, model_id, cropped_id, annotation):



    sql_engine = create_engine(mysqlurl, pool_recycle=3600)
    sql_inspector = inspect(sql_engine)

    n_grid_height, n_grid_width, maximum_in_grid = get_grid_info(model_id)

    yx = np.array(annotation)[:, 3:1:-1]
    yx = yx * np.array([[n_grid_height, n_grid_width]]) + np.array([[n_grid_height, n_grid_width]])
    yx = np.floor(yx/2).astype(int)
    yx = torch.from_numpy(yx)

    # y_pres = tf.scatter_nd(yx, tf.ones(yx.shape[:1]), (n_grid_height, n_grid_width)).numpy()
    y_pres = torch.zeros((n_grid_height, n_grid_width))
    y_pres = y_pres.index_put_(
        [
            yx[:, 0],
            yx[:, 1]
        ],
        torch.ones(yx.shape[:1]),
        accumulate=True
    )

    y_pres_np = y_pres.numpy()



    annotation = json.dumps(annotation, cls=NumpyEncoder)
    y_pres_str = json.dumps(y_pres_np, cls=NumpyEncoder)

    data = pd.DataFrame({
        "username": username,
        "model_id": model_id,
        "cropped_id": cropped_id,
        "annotation": annotation,
        "y_pres": y_pres_str
    }, index=[0])


    annotation_id = check_annotation(model_id, cropped_id, "manual")
    # print("annotation_id: ", annotation_id)
    if annotation_id == -1:

        table = "cellist_annotation_manual"
        check = sql_inspector.has_table(table)
        if check:
            data.to_sql(table, con=sql_engine, index=False, if_exists="append")
        else:
            data.to_sql(table, con=sql_engine, index=False, if_exists="replace")
    else:
        res = update_annotation(annotation_id, annotation, y_pres_str, "manual")
        # print(res)

    model_2d = ModelD2Init(model_id)
    model_2d.update_dataset(cropped_id, y_pres)

    return True, model_id, "Success"


def initialize_with_algorithm(username, model_id, algorithm):

    sql_engine = create_engine(mysqlurl, pool_recycle=3600)
    sql_inspector = inspect(sql_engine)

    
    n_grid_height, n_grid_width, maximum_in_grid = get_grid_info(model_id)
    

    # data = pd.DataFrame({
    #     "username": username,
    #     "model_id": model_id,
    #     "cropped_id": cropped_id,
    #     "annotation": json.dumps(annotation, cls=NumpyEncoder)
    # }, index=[0])


    query = pd.read_sql(
        f"\
        SELECT t1.cropped_id, t1.path, t3.object_min, t3.object_max \
        FROM \
             cellist.cellist_cropped t1  LEFT JOIN cellist.cellist_models t3 \
                    ON t1.model_id=t3.model_id \
        WHERE t1.model_id='{model_id}'\
        ",
        con=sql_engine
    )
    query = query.values


    # print(query)

    model = models.Cellpose(gpu=True, model_type='nuclei')
    model_2d = ModelD2Init(model_id)

    for i, (cropped_id, path, object_min, object_max) in enumerate(query):

        object_size = np.log(object_min) + np.log(object_max)
        object_size /= 2
        object_size = np.exp(object_size)

        img = io.imread(path)

        # if len(img.shape) == 3:
        #     gray = img[..., 0]
        # else:
        #     gray = img

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

        try:
            # masks, flows, styles, diams = model.eval(img, diameter=object_size, channels=[0, 0])
            masks, flows, styles, diams = model.eval(gray, channels=[0, 0])

            print("diams: \n", diams)

            io.imsave("cellpose_test.png", masks)
        except Exception as e:
            print(e)
            masks = np.zeros_like(gray)

        filename, _ = os.path.splitext(path)
        filename_mask = f"{filename}-masks-{algorithm}.png"
        io.imsave(filename_mask, masks)


        object_num = masks.max()

        # continue if no cells
        if object_num == 0:
            continue

        height, width = gray.shape
        x = np.linspace(-1+1/width, 1-1/width, width, endpoint=True)
        y = np.linspace(-1+1/height, 1-1/height, height, endpoint=True)
        xx, yy = np.meshgrid(x, y)


        annotation = []
        for j in range(1, object_num+1):
            mask_obj = masks == j
            xx_obj = xx[mask_obj]
            yy_obj = yy[mask_obj]

            x_min = xx_obj.min()
            x_max = xx_obj.max()

            y_min = yy_obj.min()
            y_max = yy_obj.max()

            sx = (x_max - x_min) / 2
            sy = (y_max - y_min) / 2
            x_shift = xx_obj.mean()
            y_shift = yy_obj.mean()

            # if max(sx*width, sy*height) < object_min or min(sx*width, sy*height) > object_max:
            #     continue

            annotation.append([sx, sy, x_shift, y_shift])



        y_pres = torch.zeros((n_grid_height, n_grid_width))

        if len(annotation) > 0:

            yx = np.array(annotation)[:, 3:1:-1]
            # yx = yx * np.array([[n_grid_height, n_grid_width]]) + np.array([[n_grid_height, n_grid_width]])
            # yx = np.floor(yx/2).astype(int)
            yx = (yx + 1) * np.array([[n_grid_height, n_grid_width]]) / 2
            yx = np.floor(yx).astype(int)
            yx = torch.from_numpy(yx)

            # y_pres = tf.scatter_nd(yx, tf.ones(yx.shape[:1]), (n_grid_height, n_grid_width)).numpy()
            
            y_pres = y_pres.index_put_(
                [
                    yx[:, 0],
                    yx[:, 1]
                ],
                torch.ones(yx.shape[:1]),
                accumulate=True
            )

        y_pres_np = y_pres.numpy()

        annotation = json.dumps(annotation, cls=NumpyEncoder)
        y_pres_str = json.dumps(y_pres_np, cls=NumpyEncoder)


        data = pd.DataFrame({
            "username": username,
            "model_id": model_id,
            "cropped_id": cropped_id,
            "annotation": annotation,
            "y_pres": y_pres_str,
            "algorithm": algorithm
        }, index=[i])

        annotation_id = check_annotation(model_id, cropped_id, "algorithm")
        if annotation_id == -1:

            table = "cellist_annotation_algorithm"
            check = sql_inspector.has_table(table)
            if check:
                data.to_sql(table, con=sql_engine, index=False, if_exists="append")
            else:
                data.to_sql(table, con=sql_engine, index=False, if_exists="replace")

        else:
            res = update_annotation(annotation_id, annotation, y_pres_str, "algorithm")
            # print(res)


        model_2d.update_dataset(cropped_id, y_pres)

    del model

    gc.collect()

    torch.cuda.empty_cache()

    return True, model_id, "Success"


def create_model_per_se(username, model_id, algorithm, images, slice_height, slice_width):

    update_model_images(model_id, images)


    crop_images_daemon(model_id, images, slice_height, slice_width)


    initialize_with_algorithm(username, model_id, algorithm)


    model_2d = ModelD2Init(model_id)
    
    # svi = model_2d.gen_svi_model()
    # model_2d.train()


    return model_2d

class MainHandler(tornado.web.RequestHandler):
    def get(self):

        self.render("templates/cellist.html", title="cellist")

class D3Handler(tornado.web.RequestHandler):
    def get(self):

        self.render("templates/cellist_3d.html", title="cellist 3D")


class TestHandler(tornado.web.RequestHandler, Dicton):
    def get(self, arg):
        print(arg)

class UploadHandler(tornado.web.RequestHandler):
    def post(self, uuid):

        sql_engine = create_engine(mysqlurl, pool_recycle=3600)
        sql_inspector = inspect(sql_engine)

        root_upload = os.path.join("data/uploads", uuid)

        os.makedirs(root_upload, exist_ok=True)


        # file1 = self.request.files[list(self.request.files.keys())[0]][0]
        # original_fname = file1['filename']
        # extension = os.path.splitext(original_fname)[1]
        # fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
        # final_filename= fname+extension
        # output_file = open("data/uploads/" + final_filename, 'wb')
        # output_file.write(file1['body'])
        # self.finish("file" + final_filename + " is uploaded")

        
        # 
        fname_all = []

        # print(len(self.request.files))

        uuids = []        
        paths = []

        for uuid, value  in self.request.files.items():
            # print("image_id", uuid)

            file = value[0]
            original_fname = file['filename']
            extension = os.path.splitext(original_fname)[1]
            fname = uuid + "@" + original_fname
            fname_all.append(fname)
            path_file = os.path.join(root_upload, fname)


            output_file = open(path_file, 'wb')
            output_file.write(file['body'])

            uuids.append(uuid)
            paths.append(path_file)
            # paths.append(os.path.abspath(path_file))

            
        data = pd.DataFrame({"image_id": uuids, "path": paths})
        data["source"] = "Upload"
            
        table = "cellist_images"
    
        check = sql_inspector.has_table(table)
        if check:
            data.to_sql(table, con=sql_engine, index=False, if_exists="append")
        else:
            data.to_sql(table, con=sql_engine, index=False, if_exists="replace")


        self.finish({"status": True, "msg": "Successfully uploaded <br/>" + "<br/>".join(fname_all)})
        # self.finish("haha")







class LoadModelHandler(tornado.web.RequestHandler):

    def post(self, *args, **kargs):
        model_id = self.get_argument("model_id", None)
        cur = self.get_argument("cursor", 0)

        try:
            cur = int(cur)
        except Exception as e:
            error_msg = "Invalid cursor. "

            raise e


        conn = pm.connect(
            **mysqlconfig
        )
        conn.select_db("cellist")
        cursor = conn.cursor()

        sql = f" \
            SELECT \
                t1.cropped_id, t1.path, t1.pos_x, t1.pos_y\
            FROM \
                cellist_model_images_rel t0 LEFT JOIN cellist_cropped t1 \
            ON \
                t0.image_id = t1.image_id \
            WHERE \
                t0.model_id = \'{model_id}\'\
            ORDER BY\
                t1.cropped_id \
            LIMIT \
                {cur}, 1 \
            "
        print(sql)

        status = cursor.execute(sql)

        # print("status: ", status)

        result = cursor.fetchone()


        # print("result: ", result)

        if result is not None:
            cropped_id = result[0]
            cropped_path = result[1]
        

            # sql_engine = create_engine(mysqlurl, pool_recycle=3600)
            # sql_inspector = inspect(sql_engine)

            img = io.imread(cropped_path).astype("float32").reshape(-1, 256*256)
            # img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img)

            model = ModelD2Init(model_id)
            # model.load_svi()
            z_where, z_where_inv, z_pres = model.predict(img)

            # print("model_id: ", model_id, "cursor: ", cur)

            self.write(json.dumps({
                "status": True,
                "model_id": model_id,
                "cursor": cur,
                "image_id": cropped_id,
                "image_path": cropped_path,
                "z_where_inv": z_where_inv[0],
                "z_pres": z_pres[0].astype(int),
                "msg": "Success"
            }, cls=NumpyEncoder))

        else:
            self.write(json.dumps({"status": False, "error_msg": "No images found in cuurrent cursor. "}))

class EchoWebSocket(tornado.websocket.WebSocketHandler):
    

    def open(self, ws_uuid):
        print("ws_uuid: ", ws_uuid)

        print("WebSocket opened. ")


    @gen.coroutine
    def on_message(self, message):
        # print(u"You said: " + message)
        # print(message)

        data_json = json.loads(message)

        if data_json["data_type"] == "create":
            status, model_id, msg = create_model_info(data_json["arguments"])
            username = data_json["arguments"]["username"]
            algorithm = data_json["arguments"]["based_on_algorithm"]
            images = data_json["arguments"]["local_images_uuid"]
            slice_width = data_json["arguments"]["slice_width"]
            slice_height = data_json["arguments"]["slice_height"]

            

            # cropped2dataset(model_id)
            # future = thread_pool.submit(crop_images, model_id).add_done_callback(worker_callbacks)

            # initialize_with_algorithm(username, model_id, algorithm)

            # create_model_per_se(username, model_id, algorithm)
            future = thread_pool.submit(
                create_model_per_se, 
                username, model_id, algorithm, 
                images, slice_height, slice_width).add_done_callback(worker_callbacks)

            self.write_message(json.dumps({
                "status": status,
                "model_id": model_id,
                "data_type": "create",
                "msg": msg
                }))


        elif data_json["data_type"] == "update":
            username = data_json["username"]
            cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            annotation = data_json["z_where"]

            save_manual_annotation(username, model_id, cropped_id, annotation) 


            self.write_message({"status": True, "msg": "Success"})

        elif data_json["data_type"] == "initialize":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["based_on_algorithm"]
            algorithm = data_json["algorithm"]


            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  
            thread_pool.submit(initialize_with_algorithm, username, model_id, algorithm).add_done_callback(worker_callbacks)

            self.write_message({"status": True, "msg": "Success"})

        elif data_json["data_type"] == "pretrain":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["arguments"]["based_on_algorithm"]

            model = ModelD2Pretrain(model_id)
            
            print("model.training: ", model.training)
            print("model.enable_train: ", model.enable_train)
            if not model.training:

                model.enable_train = True
                # model.train()
                thread_pool.submit(model.train).add_done_callback(worker_callbacks)


            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  

            self.write_message({"status": True, "msg": "Success"})

        elif data_json["data_type"] == "pretrain-stop":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["arguments"]["based_on_algorithm"]

            model = ModelD2Pretrain(model_id)
            model.enable_train = False

            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  

            self.write_message({"status": True, "msg": "Success"})


        elif data_json["data_type"] == "train":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["arguments"]["based_on_algorithm"]

            model = ModelD2Init(model_id)
            
            print("model.training: ", model.training)
            print("model.enable_train: ", model.enable_train)
            if not model.training:

                model.enable_train = True
                # model.train()
                thread_pool.submit(model.train).add_done_callback(worker_callbacks)


            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  

            self.write_message({"status": True, "msg": "Success"})

        elif data_json["data_type"] == "train-stop":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["arguments"]["based_on_algorithm"]

            model = ModelD2Init(model_id)
            model.enable_train = False

            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  

            self.write_message({"status": True, "msg": "Success"})


        elif data_json["data_type"] == "reset":
            username = data_json["username"]
            # cropped_id = data_json["image_uuid"]
            model_id = data_json["model_id"]
            # annotation = data_json["z_where"]
            # algorithm = data_json["arguments"]["based_on_algorithm"]

            model = ModelD2Init(model_id)
            model.enable_train = False
            model.reset_svi()

            # save_manual_annotation(username, model_id, cropped_id, annotation)          
            # initialize_with_algorithm(username, model_id, algorithm)  

            self.write_message({"status": True, "msg": "Success"})



    def on_close(self):
        print("WebSocket closed. ")








def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/3d", D3Handler),
        (r"/load_model/.*", LoadModelHandler),
        (r"/upload/(.*)", UploadHandler),
        (r"/test/(.*)", TestHandler),
        (r"/websocket/(.*)", EchoWebSocket),
        (r"/(favicon.ico)", tornado.web.StaticFileHandler, {"path": "statics/logo"}),
        (r"/bootstrap/(.*)", tornado.web.StaticFileHandler, {"path": "statics/node_modules/bootstrap/dist"}),
        (r"/statics/(.*)", tornado.web.StaticFileHandler, {"path": "statics"}),
        (r"/data/(.*)", tornado.web.StaticFileHandler, {"path": "data"}),
        # (r"/3d-model-element/(.*)", tornado.web.StaticFileHandler, {"path": "statics/3d-model-element"}),
        # (r"/3d-force-graph/(.*)", tornado.web.StaticFileHandler, {"path": "statics/3d-force-graph"}),
    ], autoreload=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8887)
    tornado.ioloop.IOLoop.current().start()
