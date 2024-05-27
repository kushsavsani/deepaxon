from keras.utils import normalize, to_categorical #keras dependencies for normalizing and categorizing datasets
import os #to work with files and directories
import cv2 #to work with images
import numpy as np
from sklearn.preprocessing import LabelEncoder #to relabel masks into 0,1,2
from sklearn.model_selection import train_test_split #create 70/30 train test split
from model import deepaxon_plusplus_model #model
from patch import batch_patch #create patches
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

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

def custom_generator(images, masks, batch_size):
    datagen.fit(images)
    generator = datagen.flow(images, masks, batch_size=batch_size, seed=42)
    while True:
        augmented_images, augmented_masks = next(generator)
        images[:len(augmented_images)] = augmented_images
        masks[:len(augmented_masks)] = augmented_masks
        yield images, masks
        
def save_image(image_array, filepath):
    """
    Save a NumPy array as an image file.

    Parameters:
        image_array (numpy.ndarray): The NumPy array representing the image.
        filepath (str): The file path where the image will be saved.
    """
    # Ensure image_array is of uint8 type
    if image_array.dtype != np.uint8:
        image_array = np.uint8(image_array)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)

    # Save the image
    image.save(filepath)
        
dir_path = r"D:\Research\Isaacs Lab\DeepAxon Project Docs\test"
batch_patch(dir_path)

#get the paths to the image and mask patches
image_patch_path = os.path.join(dir_path, 'images', 'cropped', 'patches')
mask_patch_path = os.path.join(dir_path, 'masks', 'cropped', 'patches')

#get the images and masks as numpy arrays
train_images = normalize(get_images(image_patch_path), axis=1)
train_masks = base_label(get_images(mask_patch_path))

#add another dimension to the arrays for training purposes
train_images = np.expand_dims(train_images, axis=3)
train_masks = np.expand_dims(train_masks, axis=3)

# # Initialize ImageDataGenerator for augmentation
# data_gen_args = dict(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest') 

# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# seed = 42  # Seed for reproducibility

# # Set the same seed for both generators to ensure the same augmentation for images and masks
# image_datagen.fit(train_images, augment=True, seed=seed)
# mask_datagen.fit(train_masks, augment=True, seed=seed)

# # Number of augmented images to generate
# num_augmented_images = 10

# # Generate augmented images and masks
# image_generator = image_datagen.flow(train_images, seed=seed, batch_size=num_augmented_images)
# mask_generator = mask_datagen.flow(train_masks, seed=seed, batch_size=num_augmented_images)

# # Concatenate augmented images and masks to the existing arrays
# for i in range(num_augmented_images):
#     augmented_image = image_generator.next()
#     augmented_mask = mask_generator.next()
    
#     train_images = np.concatenate((train_images, augmented_image), axis=0)
#     train_masks = np.concatenate((train_masks, augmented_mask), axis=0)
    
# img1 = train_images[0,:,:,0]
# img2 = train_masks[0,:,:,0]

# fig, axes = plt.subplots(1, 2)

# # Display the first image
# axes[0].imshow(img1, cmap='gray')  # Assuming it's a grayscale image
# axes[0].axis('off')  # Turn off axis
# axes[0].set_title('First Image')

# # Display the second image
# axes[1].imshow(img2, cmap='gray')  # Assuming it's a grayscale image
# axes[1].axis('off')  # Turn off axis
# axes[1].set_title('Second Image')

# plt.show()

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # rotation range in degrees
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    shear_range=0.1,  # shear intensity
    zoom_range=0.1,  # zoom range
    horizontal_flip=True,  # horizontal flip
    vertical_flip=True,  # vertical flip
    fill_mode='nearest'  # filling mode
)

# Set seed for reproducibility
seed = 42
np.random.seed(seed)

# Set paths to save augmented images and masks
augmented_image_path = os.path.join(dir_path, 'images', 'cropped', 'augmented_patches')
augmented_mask_path = os.path.join(dir_path, 'masks', 'cropped', 'augmented_patches')

# Create directories if they don't exist
os.makedirs(augmented_image_path, exist_ok=True)
os.makedirs(augmented_mask_path, exist_ok=True)

# Generate augmented images and masks
augment_generator = datagen.flow(train_images, train_masks, seed=seed)
num_augmented_images = 10  # Set the number of augmented images you want to generate
for i in range(num_augmented_images):
    augmented_images, augmented_masks = augment_generator.next()
    
    # Save augmented images and masks
    image_filename = f"augmented_image_{i}.png"
    mask_filename = f"augmented_mask_{i}.png"
    
    image_filepath = os.path.join(augmented_image_path, image_filename)
    mask_filepath = os.path.join(augmented_mask_path, mask_filename)
    
    # Save images
    # Assuming you have a function to save numpy arrays as images, e.g., save_image(image, filepath)
    save_image(augmented_images[0], image_filepath)
    save_image(augmented_masks[0], mask_filepath)
