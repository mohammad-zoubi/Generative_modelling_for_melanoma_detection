""" Make video of images """

import os
import moviepy.video.io.ImageSequenceClip

#########################################
#### TO MAKE MONTAGE VIDEO OF IMAGES ####
#########################################

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
