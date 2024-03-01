'''
-------------------------------- DEEPAXON --------------------------------
patch is used to create a folder of 256x256 patches of each cropped training image to be used to train a model
batch_patch should be used on a training folder with the following structure:
training
|__images
   |__image1.png
   |__image2.png
   |__ . . .
|__masks
   |__image1.png
   |__image2.png
   |___ . . .
   
and makes it look like:
training
|__images
   |__cropped
      |__patches
         |__image1_01.png
         |__image1_02.png
         |__ . . .
      |__image1.png
      |__image2.png . . .
   |__image1.png
   |__image2.png . . .
|__masks
   |__cropped
      |__patches
         |__image1_01.png
         |__image1_02.png
         |__ . . .
      |__image1.png
      |__image2.png . . .
   |__image1.png
   |__image2.png . . .
'''

from patchify import patchify #for simple patch making
import cv2 #for image processing
import os #to work with files and directories
from resize import dir_crop #dir_crop function from DeepAxon.resize

def patch(image_path):
    '''
    Take input of a single image path. Save 256x256 patches to a subfolder called 'patches' for the single image.
    The patches will be B&W png images regardless of the initial picture format.
    
    :param image_path: A path (string or object) that points to a single image to be turned into patches
    
    :returns: Nothing
    '''
    
    #extract information about the image name and path
    image_name = os.path.basename(image_path).split('.')[0] #get the name of the image. eg. 'train/images/cropped/image1.png' --> 'image1'
    image_root = os.path.dirname(image_path) #get the path to the directory the image is in. eg. 'train/images/cropped/image1.png' --> 'train/image/cropped/'
    
    #make a subfolder named 'patches' in the root folder of the image
    patch_dir = os.path.join(image_root,'patches')
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    image = cv2.imread(image_path)
    print(image_path)
    if len(image.shape) == 3 and image.shape[2] == 3:
       image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts the image to black-and-white but does not save the image anywhere. just used for processing
    
    patches_img = patchify(image, (256,256), step=256) #using patchify to make patches
    
    #saves all patches in the new patches folder
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):    
            single_patch_img = patches_img[i,j,:,:]
            patch_path = os.path.join(patch_dir, image_name+'_'+str(i)+str(j)+'.png')
            cv2.imwrite(patch_path, single_patch_img)

def dir_patch(dir_path):
    '''
    Takes input of a folder of images. In the proper setup for patch, this program will be run for either the images or masks folder.
    Turns every photo in the folder into patches and saves the patches in a new 'patches' folder.
    Uses DeepAxon.patch function patch() to generation patches.
    
    :param dir_path: A path (string or object) that points to a folder of images to convert every image in the folder into patches
    
    :returns: Nothing
    '''
    
    #goes through every image in the folder and runs the DeepAxon.patch function patch()
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        patch(image_path)
        
def batch_patch(dir_path):
    '''
    Takes input of a training folder that follows the following setup:\n
    training\n\timages\n\t\timage1.png\n\n\t\timage2.png\n\tmasks\n\t\timage1.png\n\n\t\timage2.png
    
    The folder called 'training' should be the parameter. For both the images and masks subfolders, this function will create a folder of cropped images.
    Inside the folder of copped images, a subfolder of 256x256 patches will be made. The new folder structure will be as follows:\n
    training\n\timages\n\t\tcropped\n\t\t\tpatches\n\t\t\t\timage1_01.png\n\n\t\t\t\timage1_02.png . . .\n\t\t\timage1.png\n\n\t\t\timage2.png
    \n\t\timage1.png\n\n\t\timage2.png\n\tmasks . . .
    
    :param dir_path: A path (string or object) that points to a training folder that follows the above structuring
    
    :returns: Nothing
    '''
    
    #make variables for the path the the images and masks folders
    image_path = os.path.join(dir_path, 'images')
    mask_path = os.path.join(dir_path, 'masks')
    
    #make the cropped folders with cropped images and masks
    image_path = dir_crop(image_path)
    mask_path = dir_crop(mask_path)
    
    #make patches for the cropped images and masks
    dir_patch(image_path)
    dir_patch(mask_path)