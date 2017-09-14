class NetworkGenerator:

    def __init__(self):
        self.author = 'ACN'



    def get_vggpre_net(self, pic_size)    : 
                
        import keras
        from keras import applications
        from keras.applications.vgg16 import VGG16
        from keras.models import Sequential, Model 
        from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
        from keras import backend as k 

        model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))

        # Freeze the layers which you don't want to train. Here I am freezing first five layers.
        for layer in model.layers:
            layer.trainable = False

            #Adding custom Layers 
        x = model.output
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation="relu")(x)
        predictions = Dense(3, activation="softmax")(x)

            # creating the final model 
        my_model = Model(input = model.input, output = predictions)

        return my_model, 'VGG16-pre-trained'
        
    def get_vgg_net(self, pic_size)    :
        
        """ a VGG16-like model from the example networks of keras
        https://keras.io/applications/#vgg16
        """
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        
        

        feat_width=3 # width of the first conv. filters
        number_of_feat=32 # number of first conv. filters
        im_width=224
        im_length=224
        model = Sequential()
        # input: 256x256 images with 3 channels 
        # this applies number_of_feat convolution filters of size feat_widthxfeat_width each.
        model.add(Conv2D(number_of_feat, (feat_width,feat_width), activation='relu', input_shape=(im_length, im_width, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(Dropout(0.5))#
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3,activation = 'softmax'))
        return model, 'VGG16'


##****networks below are suitable for tflearn        
    def get_alex_net(self, pic_size):

        """ AlexNet.
        Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
        References:
            - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
            Classification with Deep Convolutional Neural Networks. NIPS, 2012.
            - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
        Links:
            - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
            - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
        """

        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.normalization import local_response_normalization
        from tflearn.layers.estimator import regression

        # Building 'AlexNet'
        network = input_data(shape=[None, pic_size[0], pic_size[1], 3])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 3, activation='softmax')
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network, 'AlexNet'

    def get_convnet(self, pic_size):

        """ This is a convolutional network for 3-class classification, inspired by the 
        convolutional network applied to CIFAR-10 dataset classification task.
        References:
            Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
        Links:
            [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
        """
        
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D        
        
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation = 'relu', input_shape= (224, 224, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))     
        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(32, activation = 'relu'))
        
        model.add(Dropout(0.5))
        model.add(Dense(3, activation = 'softmax'))
       
        return model, 'ConvNet'

    def get_highway(self, pic_size):

        """ Convolutional Neural Network for MNIST dataset classification task.
        References:
            Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
            learning applied to document recognition." Proceedings of the IEEE,
            86(11):2278-2324, November 1998.
        Links:
            [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
        """

        #from __future__ import division, print_function, absolute_import

        import tflearn
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import highway_conv_2d, max_pool_2d
        from tflearn.layers.normalization import local_response_normalization, batch_normalization
        from tflearn.layers.estimator import regression

        # Building convolutional network
        network = input_data(shape=[None,pic_size[0], pic_size[1], 3], name='input')
        # highway convolutions with pooling and dropout
        for i in range(3):
            for j in [3, 2, 1]:
                network = highway_conv_2d(network, 16, j, activation='elu')
            network = max_pool_2d(network, 2)
            network = batch_normalization(network)

        network = fully_connected(network, 128, activation='elu')
        network = fully_connected(network, 256, activation='elu')
        network = fully_connected(network, 3, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')

        return network, 'highway'
