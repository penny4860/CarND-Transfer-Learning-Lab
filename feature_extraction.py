import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.layers import Input, Flatten, Dense
from keras.models import Model


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file',
                    'vgg_cifar10_100_bottleneck_features_train.p',
                    "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file',
                    'vgg_cifar10_bottleneck_features_validation.p',
                    "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs',
                    1,
                    "number of epoches")
flags.DEFINE_integer('batch_size',
                    32,
                    "batch size")



def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    import numpy as np
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # define model
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    from keras.datasets import cifar10
    from sklearn.model_selection import train_test_split
    
    tf.app.run()


    # 1) cifar10
    # Train on 1000 samples, validate on 10000 samples
    # 100 samples * 10 classes = 1000 training samples
    # VGG : loss: 0.0308 - acc: 1.0000 - val_loss: 0.9029 - val_acc: 0.7463
    # Inceltion : loss: 0.0129 - acc: 1.0000 - val_loss: 1.1957 - val_acc: 0.6625
    # resnet : loss: 0.0102 - acc: 1.0000 - val_loss: 0.8966 - val_acc: 0.7361

    # 2) Traffic Sign
    # Train on 4300 samples, validate on 12939 samples
    # 100 samples * 43 classes = 4300 training samples
    # VGG
    # Epoch 50/50
    # 4300/4300 [==============================] - 0s - loss: 0.0873 - acc: 0.9958 - val_loss: 0.4368 - val_acc: 0.8666
    # Inception
    # Epoch 50/50
    # 4300/4300 [==============================] - 0s - loss: 0.0276 - acc: 1.0000 - val_loss: 0.8378 - val_acc: 0.7519
    # ResNet
    # Epoch 50/50
    # 4300/4300 [==============================] - 0s - loss: 0.0332 - acc: 1.0000 - val_loss: 0.6146 - val_acc: 0.8108





