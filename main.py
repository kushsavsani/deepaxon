from segment import segment
from morphometrics import get_morphometrics
from morphometrics import save_morphometrics
from train import train_model
import os



# '''--------------------------------TRAIN MODEL------------------------------'''
# dir_path = r"D:\Research\Isaacs Lab\DeepAxon\training"
# model_path = r"D:\Research\Isaacs Lab\DeepAxon\models"
# model_name = 'model_10img_16bs-200epoch'

# train_model(dir_path, model_path, model_name, batch_size=16, epochs=200)
# '''-----------------------------END TRAIN MODEL------------------------------'''






# '''---------------------------------SEGMENT ONE IMAGE-------------------------------------'''
# img_path = r"D:\Research\Isaacs Lab\23G-15\23G-15_03\23G-15_40X_03_0002\23G-15_40X_03_0002.tif"
# model_path = r"D:\Research\Isaacs Lab\DeepAxon\models\model_10img_16bs-200epoch.keras"
# output_path = r"D:\Research\Isaacs Lab\DeepAxon\test"

# segment(img_path, model_path, output_path)
# '''-------------------------------END SEGMENT ONE IMAGE-------------------------------------'''








# '''---------------------SEGMENT ALL IMAGES IN A FOLDER------------------------------'''
# dir_path = r"E:\Research\Isaacs Lab\DeepAxon Project Docs\test\images"
# model_path = r"D:\Research\Isaacs Lab\DeepAxon\models\model_10img_16bs-200epoch.keras"
# output_path = r"D:\Research\Isaacs Lab\DeepAxon\test"

# for img_name in os.listdir(dir_path):
#     img_path = os.path.join(dir_path,img_name)
#     segment(img_path, model_path, output_path)
# '''--------------------END SEGMENT ALL IMAGES IN A FOLDER--------------------------'''







# '''------------------GET MORPHOMETRICS FOR ONE IMAGE--------------------------------'''
# img_path = r"D:\Research\Isaacs Lab\23G-15\23G-15_03\23G-15_40X_03_0002\23G-15_40X_03_0002.tif"
# output_path = r"D:\Research\Isaacs Lab\DeepAxon\test"
# output_name = 'morphometrics.xlsx'

# morph_df = get_morphometrics(img_path)
# save_morphometrics(morph_df, output_path, output_name)
# '''----------------END GET MORPHOMETRICS FOR ONE IMAGE------------------------------'''







# '''-----------------------MORPHOMETRICS FOR ALL IMAGES IN A FOLDER------------------------'''
# dir_path = r"E:\Research\Isaacs Lab\DeepAxon Project Docs\test\images"
# output_path = r"D:\Research\Isaacs Lab\DeepAxon\test"

# for img_name in os.listdir(dir_path):
#     img_path = os.path.join(dir_path, img_name)
#     output_name = img_name.split('.')[0] + '_morphometrics.xlsx'
#     morph_df = get_morphometrics(img_path)
#     save_morphometrics(morph_df, output_path, output_name)
# '''------------------END MORPHOMETRICS FOR ALL IMAGES IN A FOLDER-------------------------'''