""" Make video of images """

import os
import click
import moviepy.video.io.ImageSequenceClip
# image_folder_path='/SAM256/models/256_nopreprocessing/00001--cond-auto1-bgcfnc-resumecustom/' # sam
# image_folder_path='/ISIC256/models/00000-ISIC256_benign-auto1-bgcfnc/' # ISIC
# image_folder_path='/data/stylegan2-ada-pytorch/para_imgs_SAM/' # SAM
# fps=1
# print(os.listdir(image_folder_path))

# @click.command()
# @click.pass_context
# @click.option('--image_folder_path',type=str, help='where are the images', required=True)
# @click.option('--fps', type=int, help='nr of fps', required=True)
# @click.option('--outdir', type=str, help='out video path', required=True)

# image_files = [os.path.join(image_folder_path,img)
#                for img in os.listdir(image_folder_path)
#                if (img.endswith(".jpg") and img[:6] == 'fakes0')]
def make_video(image_folder_path, fps, outdir):
    image_files = [os.path.join(image_folder_path,img)
                for img in os.listdir(image_folder_path)
                if (img.endswith(".jpg") or img.endswith(".png"))]
    # print(image_files)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(outdir)


image_folder_path = '/data/generated_uncond/generated_imgs_ISIC_mal/directed_imgs/'
fps = 3
outdir = '/data/stylegan2-ada-pytorch/directed_imgs_f15f4.mp4'
make_video(image_folder_path, fps, outdir)

# if __name__ == "__main__":
#     make_video() 
