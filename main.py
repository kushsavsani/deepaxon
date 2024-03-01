from segment import segment
# from morphometrics import get_morphometrics
# from train import train_model
# import os



# '''--------------------------------TRAIN MODEL------------------------------'''
# dir_path = r"D:\Research\Isaacs Lab\DeepAxon\training"
# model_path = r"D:\Research\Isaacs Lab\DeepAxon\models"
# model_name = 'model_10img_16bs-200epoch'

# train_model(dir_path, model_path, model_name, batch_size=16, epochs=200)



'''-----------------------SEGMENT IMAGE-------------------------------------'''
img_path = r"D:\Research\Isaacs Lab\23G-15\23G-15_03\23G-15_40X_03_0002\23G-15_40X_03_0002.tif"
model_path = r"D:\Research\Isaacs Lab\DeepAxon\models\model_10img_16bs-200epoch.keras"
output_path = r"D:\Research\Isaacs Lab\DeepAxon\test"

segment(img_path, model_path, output_path)