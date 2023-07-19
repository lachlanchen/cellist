import cv2
import numpy as np

def image_slice(image, color_depth=8**2 - 1, model_shape=(256, 256), n_channels=3):

   # print("image.shape: ", image.shape)

   model_height = model_shape[0]
   model_width = model_shape[1]
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


    
   image_holder_overlap = np.zeros((height_holder+model_height, width_holder+model_width, n_channels), dtype=image.dtype) + ret
   image_holder_overlap[model_height_half:height+model_height_half, model_width_half:width+model_width_half] = image

   images_overlap = np.lib.stride_tricks.as_strided(image_holder_overlap, 
         shape=(n_window_height+1, n_window_width+1, model_height, model_width, n_channels), 
         strides=(model_height*(n_window_width+1)*model_width*n_channels*elem_len, model_width*n_channels*elem_len, (n_window_width+1)*model_width*n_channels*elem_len, n_channels*elem_len, elem_len))
   images_overlap = images_overlap.reshape(-1, model_height, model_width, n_channels)


   return images, images_overlap, (height, width)





def image_stitch(images, images_overlap, original_shape=(256, 256), model_shape=(256, 256)):
   
   model_height = model_shape[0]
   model_width = model_shape[1]
   model_height_half = model_height // 2
   model_width_half = model_width // 2

   height = original_shape[0]
   width = original_shape[1]
   n_window_height = np.ceil(height / model_height).astype(int)
   n_window_width = np.ceil(width / model_width).astype(int)
   height_holder = (n_window_height * model_height).astype(int)
   width_holder = (n_window_width * model_width).astype(int)

   image_holder_overlap = np.zeros((height_holder+model_height, width_holder+model_width), dtype=images.dtype)
   for i in range(n_window_height+1):

         for j in range(n_window_width+1):

            image_holder_overlap[i*model_height:(i+1)*model_height, j*model_width:(j+1)*model_width] = images_overlap[i*(n_window_width+1) + j]

   image_overlap  = image_holder_overlap[model_height_half:height+model_height_half, model_width_half:width+model_width_half]


   image_holder = np.zeros((height_holder, width_holder), dtype=images.dtype)
   for i in range(n_window_height):

         for j in range(n_window_width):

            image_holder[i*model_height:(i+1)*model_height, j*model_width:(j+1)*model_width] = images[i*n_window_width + j]

            # if i > 0:
            #     image_holder[(i-1)*256+96:i*256-96, j*256:(j+1)*256] = image_overlap[(i-1)*256+96:i*256-96, j*256:(j+1)*256]

            # if j > 0:
            #     image_holder[i*256:(i+1)*256, (j-1)*256+96:j*256-96] = image_overlap[i*256:(i+1)*256, (j-1)*256+96:j*256-96]

   image = image_holder[:height, :width]

   # boarder = (image_overlap>0.5) > (image>0.5)
   # boarder = image_overlap > image
   # boarder = sp.ndimage.binary_dilation(boarder)
   # boarder = sp.ndimage.binary_dilation(boarder)
   # image = np.maximum(image, image_overlap)
   # image[boarder] = image_overlap[boarder]
   image = (image+image_overlap) / 2


   return image