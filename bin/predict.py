import os

import PIL
from PIL import Image
import numpy as np
import cv2
import sys

import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.base import Model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('model')
    args = parser.parse_args()

    model_path = args.model
    data_path = args.image_dir

    print('Loading model from ', model_path)
    model = Model().load(model_path)

    print('Model loaded')
    print(model._config)
    
    print('## Evaluating on test data##')
    prediction, labels = model.predict(data_path, return_labels=True)
    prediction_class = prediction.argmax(axis=-1)

    correct = (prediction_class == labels).sum()
    total = len(labels)

    print('Percentage correct (manual): {:.2f}, {}/{}'.format((correct / total * 100), correct, total))

    #np.save('predictions.npy', {'prediction': prediction, 'true': y_test, 'labels': labels})
    

    """
    # Legacy code

    print('Loading model from ', model_path)
    model = tf.keras.models.load_model(model_path)
    print('Model loaded')
    
    im_size = args.size
    batch_size = args.batch_size
    labels = args.classes
    label2oh = dict( (e, np.eye(len(labels))[i]) for i, e in enumerate(labels) ) 
    
    if args.preprocess == 'inceptionv3':
        preprocess = tf.keras.applications.inception_v3.preprocess_input
    elif args.preprocess == 'inception_resnetv2':
        preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input
    else:
        preprocess = None

    # TODO: Use data loading from data.py
    def load_image(img_path):
        # Load image
        try:
            image = Image.open(img_path)
        except:
            return None

        # Convert to grayscale
        if image.mode == 'RGB':
            image = image.convert('L')

        # Convert to numpy array
        image = np.array(image, dtype='float32')

        # Squeeze extra dimensions
        if len(image.shape) == 3:
            image = np.squeeze(image)

        # Resize
        if image.shape != (im_size, im_size):
            image = cv2.resize(image, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)

        # Make grayscale 3 channel input (might be able to bin this)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image[np.newaxis, :, :, :]

        # Do any image preprocessing
        if preprocess:
            image = preprocess(image)
        else:
            image /= 255
        
        return image

    def load_data(data_path):
        X = []
        y = []

        for d in os.listdir(data_path):
            for f in os.listdir(os.path.join(data_path, d)):
                img_path = os.path.join(os.path.join(data_path, d), f)
                image = load_image(img_path)
                if image is None: continue
                X.append(image)
                y.append(label2oh[d])
        return np.concatenate(X, axis=0), np.array(y)

    print('Loading data from ', data_path)
    x_test, y_test = load_data(data_path)
    print("Data loaded")

    prediction = model.predict(x_test)
    prediction_class = prediction.argmax(axis=-1)
    true_class = y_test.argmax(axis=-1)
    correct = (prediction_class == true_class).sum()
    total = len(y_test)

    print('Percentage correct (manual): {:.2f}, {}/{}'.format((correct / total * 100), correct, total))
    
    np.save('predictions.npy', {'prediction': prediction, 'true': y_test, 'labels': labels})
    """