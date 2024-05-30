import train

dir_path = input("Input the path to the training folder that holds the images and masks: ")
model_path = input("Input the path of the folder where the model will be saved: ")
model_name = input("Input the name of the model: ")
batch_size = (int) (input("Batch size (16 recommended): "))
epochs = int (input("Epochs (200 recommended): "))

train.train_model(dir_path, model_path, model_name, batch_size, epochs)