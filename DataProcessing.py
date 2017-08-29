import os, random
from PIL import Image
import numpy as np
import av
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# the libraries below are for image preprocessing

#import skimage as ski
#from skimage import data, feature
#from skimage import exposure
#from skimage.exposure import rescale_intensity

# TODO: video and data root | and 'rawpictures'
# TODO: v2image should write to a folder using resize pics in title. to avoid retraining etc.

class PreProcessor:

    def __init__(self, size_pics=(180,320)):
        self.size_pics = size_pics

#  this function is not yet used anywhere        
    def create_data(self):
        self.video_to_images(size_pics=self.size_pics)
        self.augment_data()

    def augment_data(self, image_path='./raw_pictures', augment_path='./processed_data', batch_size=10):
        
        # Define data augmentor
        dataaug = ImageDataGenerator(
            #rescale=1./255,
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.1,
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
    def video_to_images(image_path='./raw_pictures', video_path='./trimmed_videos', size_pics=(256, 256), frame_spacing=24):

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
                        img.save(image_path + '/' + folder + '/' + head + 'frame-%04d.png' % frame.index)
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
    def load_data(root_directory, size_pic=(180,320)):

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
    def balance_image_classes(maximum, augment_path='./processed_data'):
    # check if destination folder present
        if os.path.exists(augment_path):
            folders = [f.path for f in os.scandir(augment_path)]
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
                    



# this is just an experimental function to simplify
# the images to the bare features: modem outline & lamps        
    @staticmethod
    def process_array(array):
        img_r=array[:,:,0]
        img_g=array[:,:,1]
        img_b=array[:,:,2]
        images=[img_r,img_g,img_b]
        processed_img=[]
        [l,w]=img_r.shape
#what if we subtract the edgeness from blobness?
        for im in images:
            log = ski.feature.blob_log(im, max_sigma=2,threshold=.1,overlap=.1)
            edges = feature.canny(im, sigma=3)
            blurred_edges = ski.filters.gaussian(edges.astype(float),sigma=2)
            blurred_edges[blurred_edges<0]=0
            blobcentra=np.zeros((l,w))
            for index in log[:, 0:2]:
                blobcentra[index[0].astype(int),index[1].astype(int)]=1.0
            blob=ski.filters.gaussian(blobcentra.astype(float),sigma=2)
            blob_but_no_edge=blob-blurred_edges
            
            processed_img.append(blob_but_no_edge)

        narray=np.dstack((processed_img[0],processed_img[1],processed_img[2]))        
        return narray