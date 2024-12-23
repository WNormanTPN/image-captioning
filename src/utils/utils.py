import string

import numpy as np
from tqdm import tqdm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input



# Read captions file
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# Save caption descriptions to a dictionary
#id_image : ['caption 1', 'caption 2', 'caption 3',' caption 4', 'caption 5']
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


# Preprocessing text
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # add 'startseq ' and ' endseq' to each description
            desc.insert(0, 'startseq ')
            desc.append(' endseq')
            # store as string
            desc_list[i] = ' '.join(desc)
            
            
# Save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    with open(filename, 'w') as file:
        for key, desc_list in tqdm(descriptions.items(), desc="Saving descriptions"):
            for desc in desc_list:
                file.write(f"{key} {desc}\n")
                

# Preprocessing images for inception v3 model
def preprocess_image(img):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(img, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    # Convert numpy array to PIL image
    x = image.array_to_img(x[0])
    return x


# Save the processed images
def save_images(images, folder_name):
    for key, image in tqdm(images.items(), desc="Saving images"):
        filename = folder_name + key.split('/')[-1]
        image.save(filename)