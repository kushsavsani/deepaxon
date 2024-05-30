import morphometrics

img_path = input("Input the path to the image: ")
output_dir = input("Input the path to the output folder: ")
output_name = input("Input the name of the output file: ")

morph_df = morphometrics.get_morphometrics(img_path)
morphometrics.save_morphometrics(morph_df, output_dir, output_name)