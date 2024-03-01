'''
-------------------------------- DEEPAXON --------------------------------
Train a segmentation model using the DeepAxon++ model. Training folder should follow the folder structure of:

training
|__images
   |__image1.png
   |__image2.png
   |__ . . .
|__masks
   |__image1.png
   |__image2.png
   |___ . . .
   
can specifiy the name and the folder to save it in
'''

from keras.utils import normalize, to_categorical #keras dependencies for normalizing and categorizing datasets
import os #to work with files and directories
import cv2 #to work with images
import numpy as np
from sklearn.preprocessing import LabelEncoder #to relabel masks into 0,1,2
from sklearn.model_selection import train_test_split #create 70/30 train test split
from model import deepaxon_plusplus_model #model
from patch import batch_patch #create patches

def get_images(patch_path):
    '''
    Get all the images in a folder as a numpy array
    
    :param patch_path: A path (string or object) 
    
    :returns: Numpy Array; all images in the patches folder
    '''
    
    #go to every list in the patch folder and append it to the list
    img_array = []
    for image_name in os.listdir(patch_path):
        image_path = os.path.join(patch_path, image_name)
        img = cv2.imread(image_path, 0)
        img_array.append(img)

    #convert 
    return np.array(img_array)

def base_label(train_masks):
    '''
    If the manual masks are made with colors instead of 0,1,2 this function will convert it into 0,1,2.
    
    :param train_masks: Numpy Array; training mask images
    
    :returns: Numpy Array; training mask images with values being 0,1,2 instead of whatever colors it was
    '''
    
    label_encoder = LabelEncoder()
    n, h, w = train_masks.shape #get shape values to be eable to convert it back into its shape
    train_masks_flat = train_masks.reshape(-1) #make the array 1 dimensional
    train_masks_flat_encoded = label_encoder.fit_transform(train_masks_flat) #recode the labels to make them base values
    train_masks_encoded = train_masks_flat_encoded.reshape(n, h, w) #reshape the array to fit into original shape
    return train_masks_encoded

def train_model(dir_path, model_path, model_name, batch_size=16, epochs=200):
    '''
    Train and save model from training images and masks. Uses DeepAxon++ architecture
    
    :param dir_path: A path (string or object) to a training folder which had both images and masks folders
    :param model_path: The path to the directory to save the model in. Recommend making a model folder to save models in
    :param model_name: Name of the model to be save as
    
    :returns: Keras model
    '''
    
    batch_patch(dir_path) #create patches folder for training images and masks
    
    #get the paths to the image and mask patches
    image_patch_path = os.path.join(dir_path, 'images', 'cropped', 'patches')
    mask_patch_path = os.path.join(dir_path, 'masks', 'cropped', 'patches')
    
    #get the images and masks as numpy arrays
    train_images = normalize(get_images(image_patch_path), axis=1)
    train_masks = base_label(get_images(mask_patch_path))
    
    #add another dimension to the arrays for training purposes
    train_images = np.expand_dims(train_images, axis=3)
    train_masks = np.expand_dims(train_masks, axis=3)
    
    #70/30 training/testing split
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size = 0.10, random_state = 0)
    
    #convert training masks into categorical variables
    train_masks_cat = to_categorical(y_train, num_classes=3)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], 3)) #last dimension is 3 for background, axon, myelin
    
    #convert test masks into categorical variables
    test_masks_cat = to_categorical(y_test, num_classes=3)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 3)) #last dimension is 3 for background, axon, myelin
    
    #get shape of training set
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    
    model = deepaxon_plusplus_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=3) #obtain model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile architecture
    model.fit(X_train, y_train_cat, batch_size=batch_size, verbose=1, epochs=epochs, validation_data=(X_test, y_test_cat), shuffle=False) #train model
    
    model.save(os.path.join(model_path, model_name+'.keras')) #save model