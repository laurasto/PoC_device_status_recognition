import os, random
import PIL
from PIL import Image
import numpy as np

import av
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#import skimage 
#from skimage import color


#from skimage import data, feature
#from skimage import exposure
#from skimage.exposure import rescale_intensity

# TODO: video and data root | and 'rawpictures'
# TODO: v2image should write to a folder using resize pics in title. to avoid retraining etc.

#import matplotlib.pyplot as plt



class PreProcessor:

    def __init__(self, size_pics=(224, 224)):
        self.size_pics = size_pics
        
#  this function is not yet used anywhere        
    def create_data(self):
        self.video_to_images(size_pics=self.size_pics)
        self.augment_data()

    def augment_data(self, image_path='./raw_pictures', augment_path='./processed_data', batch_size=10):
        
        # Define data augmentor
        dataaug = ImageDataGenerator(
            #rescale=1./255,
            rotation_range=90,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0,
            zoom_range=[0.8,0.95],
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest")

        # check if destination folder present
        if not os.path.exists(augment_path):
            os.makedirs(augment_path)

        folders = [f.name for f in os.scandir(image_path) if f.is_dir()]
        for folder in folders:
            files = [f.path for f in os.scandir(image_path + '/' + folder)]

            # check if destination folder present
            if not os.path.exists(augment_path + '/' + folder):
                os.makedirs(augment_path + '/' + folder)

            # for each raw image
            for file in files:
                img = load_img(file)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)  # dim: (1, height, width, rbg)
                i = 0
                for batch in dataaug.flow(x, batch_size=1, save_to_dir=augment_path + '/' + folder, save_prefix='aug',
                                          save_format='png'):
                    i += 1
                    if i > batch_size:
                        break

    @staticmethod
    def video_to_images(image_path='./raw_pictures', video_path='./trimmed_videos', size_pics=(640, 360), frame_spacing=12):

        # check if destination folder present
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        folders = [folder for folder in os.listdir(video_path)]
        for folder in folders:
            files = [file for file in os.listdir(video_path + '/' + folder)]

            # check if destination folder present
            if not os.path.exists(image_path + '/' + folder):
                os.makedirs(image_path + '/' + folder)

            for file in files:
                container = av.open(video_path + '/' + folder + '/' + file)

                # resize and save frame to disk
                c = 0
                for frame in container.decode(video=0):
                    if c % frame_spacing == 0:
                        frame = frame.reformat(width=size_pics[0], height=size_pics[1])
                        img = frame.to_image()
                        head = ''.join(file.split('.')[:1])  # remove everything after punctuation
                        img.save(image_path + '/' + folder + '/' + head + 'frameH-%04d.png' % frame.index)
                    c += 1
                    
    @staticmethod
    def load_image(file):
        img = load_img(file)
        x = img_to_array(img)
        return x
        # return np.expand_dims(x, axis=0)  # dim: (1, height, width, rbg)

    @staticmethod
    def count_instances(root_dirname):
        n_images = 0
        folders = [f.path for f in os.scandir(root_dirname) if f.is_dir()]
        n_folder = len(folders)
        for folder in folders:
            n_images += len(os.listdir(folder))
        return n_images, n_folder

    @staticmethod
    def load_data(root_directory, size_pic=(224, 224)):

        folders = [f.path for f in os.scandir(root_directory) if f.is_dir()]
        n_images, n_folder = PreProcessor.count_instances(root_directory)
        data = np.zeros((n_images, size_pic[0], size_pic[1], 3), dtype=np.float32)
        target = np.zeros((n_images, n_folder), dtype=np.float32)
        i_image = 0
        i_folder = -1
        for folder in folders:
            i_folder += 1
            files = [f.path for f in os.scandir(folder)]
            for file in files:
                # Parse images
                x = PreProcessor.load_image(file)
                data[i_image, :, :, :] = x 
                target[i_image, i_folder] = 1.0
                i_image += 1

        return data, target
    # balance_image_classes takes care that there are equal amount of
    # samples from different classes and that they do not exceed a maximum
    @staticmethod
    def balance_image_classes(maximum, image_path='./raw_pictures'):
    # check if destination folder present
        if os.path.exists(image_path):
            folders = [f.path for f in os.scandir(image_path)]
            number_of_samples_per_class=[maximum]
            for folder in folders:
                n=len(os.listdir(folder))
                number_of_samples_per_class.append(n)
            max=np.min(number_of_samples_per_class)
            for folder in folders:
                while(len(os.listdir(folder))>max):
                    random_file=np.random.choice(os.listdir(folder))
                    os.remove(folder+'/'+random_file)
        return         
                    


    # use the functions below to:
    # convert an arbitrary (color) image to a RGB-png of given size 
    # combine images of segmented modems and 'appropriate' backgrounds 

    def convert_to_png(self, name, img_rows, img_cols, image_path = './raw_pictures'):
        
        folders = [f.name for f in os.scandir(image_path) if f.is_dir()]
        for folder in folders:
            files = [f.path for f in os.scandir(image_path + '/' + folder)]
            for index, file in enumerate(files):
                im = Image.open(file)
                array = np.array(im)
                array=array[:,:,:3]
                im = Image.fromarray(array)
                im = im.resize((img_rows, img_cols), PIL.Image.ANTIALIAS)
                im.save(image_path + '/' + folder + '/'+ name + str(index) + '.png')
                os.remove(file)
        return

        
    # separate a test set    
    def separate_test_set(self, train_path = './new_augmented', test_path = './test_set'):
        
        folders = [f.name for f in os.scandir(train_path) if f.is_dir()]
        for folder in folders:
            max=np.round(0.8*len(os.listdir(train_path + '/'  + folder))).astype(int)
            while(len(os.listdir(train_path + '/'  + folder))>max):
                random_file=np.random.choice(os.listdir(train_path + '/'  +folder))
                os.rename(train_path+'/'+folder+'/'+random_file, test_path + '/' + folder + '/' + random_file)

        return
 
