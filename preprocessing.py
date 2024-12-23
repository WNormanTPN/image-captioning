import os
import glob
import shutil

from src import *
from tqdm import tqdm


# Configure
input_captions_file = "./data/raw/Flickr8k.token.txt"
input_images_dir = "./data/raw/Flicker8k_Dataset"

output_captions_file = "./data/processed/captions.txt"
output_images_dir = "./data/processed/images"

train_images_file = './data/raw/Flickr_8k.trainImages.txt'
val_images_file = './data/raw/Flickr_8k.devImages.txt'
test_images_file = './data/raw/Flickr_8k.testImages.txt'


# Clean up the output directories
print('Cleaning up the output directories...')
if os.path.exists(output_images_dir):
    shutil.rmtree(output_images_dir)  # Xóa thư mục và toàn bộ nội dung
if os.path.exists(output_captions_file):
    os.remove(output_captions_file)  # Xóa file nếu tồn tại
    

# Make output directories
print('Creating output directories...')
os.makedirs(output_images_dir + '/train')
os.makedirs(output_images_dir + '/val')
os.makedirs(output_images_dir + '/test')


# Preprocess captions
print('Preprocessing captions...')
doc = load_doc(input_captions_file)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)
save_descriptions(descriptions, output_captions_file)


# Preprocess images
print('Preprocessing images...')
train_img = []
val_img = []
test_img = []

img = glob.glob(input_images_dir + '/*.jpg')

train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

for i in img: # img is list of full path names of all images
    if i[len(input_images_dir)+1:] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
    elif i[len(input_images_dir)+1:] in val_images:
        val_img.append(i)
    elif i[len(input_images_dir)+1:] in train_images:
        train_img.append(i)
        
train_img = {k: preprocess_image(k) for k in tqdm(train_img, desc='Processing train images')}
val_img = {k: preprocess_image(k) for k in tqdm(val_img, desc='Processing val images')}
test_img = {k: preprocess_image(k) for k in tqdm(test_img, desc='Processing test images')}


print ('Saving train images...')
save_images(train_img, output_images_dir + '/train/')
print('Saving val images...')
save_images(val_img, output_images_dir + '/val/')
print('Saving test images...')
save_images(test_img, output_images_dir + '/test/')