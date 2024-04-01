'''
-------------------------------- DEEPAXON --------------------------------
resize is used to crop images into a square so that when they are turned into patches there are not unused parts of the images
'''

from PIL import Image #for image-related work
import os #to work with files and directories
import numpy as np
import cv2

def cropper(image_path):
    '''
    Crops an image into a square, cutting equally from any side that has to be cut
    
    :param image_path: A path (string or object) that directs to a single image to be cropped
    
    :returns: A PIL Image object that is the centered, cropped image
    '''
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #opens the given image

    SIZE_X = image.shape[1] // 256 * 256
    SIZE_Y = image.shape[0] // 256 * 256
    
    image = Image.fromarray(image)
    cropped = image.crop((0,0,SIZE_X, SIZE_Y))

    return cropped

def dir_crop(dir_path):
    '''
    Crops every image in a directory and saves then in a subfolder called 'cropped' in the given directory
    
    :param dir_path: A path (string or object) that directs to a directory with one or more images. The directory should only contain images
    
    :returns: The path of the subfolder named 'cropped' which was made in the given folder
    '''
    
    #make a subfolder called 'cropped'
    crop_dir = os.path.join(dir_path,'cropped')
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
    
    #go through every image in the given folder and save a cropped version in the new 'cropped' subfolder
    for image_name in os.listdir(dir_path):
        
        filetype = len(image_name.split('.'))
        
        #only run for files, not folders
        if filetype == 2:
            image_path = os.path.join(dir_path, image_name)
            crop_path = os.path.join(dir_path,'cropped',image_name)
            cropped = cropper(image_path)
            cropped.save(crop_path)
    
    return crop_dir