''' places given images in one large image, preserves resolution '''

import os.path
import sys
from time import strftime
from PIL import Image

row_size = 30
margin = 3
# list_ish = os.listdir("/data/stylegan2-ada-pytorch/pres_imgs/")
# list_ish = os.listdir("/ISIC256/ISIC_pool/malignant_all/RESIZED_IMAGES/")
# image_folder='/data/stylegan2-ada-pytorch/pres_imgs'
# image_files = [os.path.join(image_folder,img)
#                for img in os.listdir(image_folder)
#                if (img.endswith(".jpg") and img[:5]=='fakes')]
def generate_montage(source, output_fn):
    filenames = os.listdir(source)
    images = [Image.open(os.path.join(source, filename)) for filename in filenames]

    width = max(image.size[0] + margin for image in images)*row_size
    height = sum(image.size[1] + margin for image in images)
    montage = Image.new(mode='RGBA', size=(width, height), color=(0,0,0,0))

    max_x = 0
    max_y = 0
    offset_x = 0
    offset_y = 0
    for i,image in enumerate(images):
        montage.paste(image, (offset_x, offset_y))

        max_x = max(max_x, offset_x + image.size[0])
        max_y = max(max_y, offset_y + image.size[1])

        if i % row_size == row_size-1:
            offset_y = max_y + margin
            offset_x = 0
        else:
            offset_x += margin + image.size[0]

    montage = montage.crop((0, 0, max_x, max_y))
    montage.save(output_fn)


# basename = strftime("Montage_frames %Y-%m-%d at %H.%M.%S.png")
# exedir = "/content/montage"
# filename = "/content/out/images"
source = "/ISIC256/ISIC_pool/malignant_all/RESIZED_IMAGES/"
generate_montage(source, "/data/stylegan2-ada-pytorch/all_ISIC_pool_imgs.png")