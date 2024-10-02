import os
from PIL import Image

def resize_images_in_directory(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if the file is an image (you can expand this with more image formats if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
            try:
                # Open an image file
                with Image.open(filepath) as img:
                    # Check if the image size is 2880x2048
                    if img.size == (2880, 2048):
                        # Resize the image to 1440x1024
                        img_resized = img.resize((1440, 1024), Image.Resampling.LANCZOS)
                        
                        # Save the resized image, overwriting the original
                        img_resized.save(filepath)
                        print(f"Resized and saved image: {filename}")
                    else:
                        print(f"Skipping {filename}: Not of size 2880x2048.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Main function to get user input
def main():
    directory = input("Enter the filepath to the directory containing images: ")
    
    if os.path.isdir(directory):
        resize_images_in_directory(directory)
    else:
        print("Invalid directory path. Please try again.")

if __name__ == "__main__":
    main()
