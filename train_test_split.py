
from PIL import Image
import numpy as np
import os

#%%

# original folder containing all the images (5653)

main_dir = r'C:\Users\Lav\Desktop\Lav\Datasets\Image datasets\Cats, dogs and other animals image dataset\resized cat images'

print(len(os.listdir(main_dir)))        # 5653 images
print('')

#%%

# splitting these 5653 images as: 5153 training images, 500 validation images

train_dir = r'C:\Users\Lav\Desktop\Lav\Datasets\Image datasets\Cats, dogs and other animals image dataset\resized cat images (training and validation)\training images'
validation_dir = r'C:\Users\Lav\Desktop\Lav\Datasets\Image datasets\Cats, dogs and other animals image dataset\resized cat images (training and validation)\validation images'

#%%

li = [4, 7, 0, 2, 6, 1, 9, 8, 5, 3]

random_arr = np.random.choice(li, size = 4, replace = False)

print(random_arr)
print('')

#%%

list_of_image_names = os.listdir(main_dir)

validation_images_array = np.random.choice(list_of_image_names, size = 500, replace = False)

# print(validation_images_array)
# print('')

# print(len(validation_images_array))
# print('')

set_of_image_names = set(list_of_image_names)
validation_images_set = set(validation_images_array)

training_images_set = set_of_image_names - validation_images_set

training_images_array = np.array(list(training_images_set))

# print(len(training_images_array))
# print('')

#%%

# saving all the images in 'training_images_array' to 'train_dir' folder

for training_image_name in training_images_array:
    training_image = Image.open(os.path.join(main_dir, training_image_name))
    training_image.save(os.path.join(train_dir, training_image_name))


#%%

# saving all the images in 'validation_images_array' to 'validation_dir' folder

for validation_image_name in validation_images_array:
    validation_image = Image.open(os.path.join(main_dir, validation_image_name))
    validation_image.save(os.path.join(validation_dir, validation_image_name))


