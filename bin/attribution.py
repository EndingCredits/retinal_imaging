import os

import PIL
from PIL import Image
import numpy as np
import cv2

import tensorflow as tf

tf.compat.v1.enable_eager_execution() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('model')
    parser.add_argument('--batch-size', default=32, help='Batch size', type=int)
    parser.add_argument('--classes', default=['ABCA4', 'USH2A'], help='List of classes', nargs='+')
    parser.add_argument('--size', type=int, default=256, help='Shape of input e.g 256 for (256,256)')
    parser.add_argument('--preprocess', choices=['inceptionv3', 'inception_resnetv2'], help='Preprocessing to perform on images')
    parser.add_argument('--new', action='store_true', help='Set if predicting on a flat folder of new data')
    args = parser.parse_args()

    model_path = args.model
    data_path = args.image_dir
    
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
    
    """ # Automatic data loading:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if not preprocess:
        rescale_val = 1./255
    else:
        rescale_val = None
    
    predictions = model.predict(
        ImageDataGenerator( rescale=rescale_val,
            preprocessing_function=preprocess).flow_from_directory(
                args.image_dir,
                target_size=(im_size, im_size),
                class_mode=None,
                batch_size=batch_size,
                shuffle=False,
            ),
        use_multiprocessing=True,
        workers=30,
        verbose=1,
    )
    """

    
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
            if d not in labels: continue
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

    
    
    import matplotlib
    matplotlib.use("Agg")
    
    import matplotlib.colors as colors
    import matplotlib.pylab as plt
    from scipy.ndimage.filters import gaussian_filter

    def interpolate_images(baseline,
                           image,
                           alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def compute_gradients(images, target_class_idx=0):
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        return tape.gradient(probs, images)

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    #@tf.function
    def integrated_gradients(baseline,
                             image,
                             target_class_idx,
                             m_steps=50,
                             batch_size=32):
        # 1. Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # Initialize TensorArray outside loop to collect gradients.    
        gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                               image=image,
                                                               alphas=alpha_batch)

            # 3. Compute gradients between model outputs and interpolated inputs.
            gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                               target_class_idx=target_class_idx)

            # Write batch indices and gradients to extend TensorArray.
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()

        # 4. Integral approximation through averaging gradients.
        avg_gradients = integral_approximation(gradients=total_gradients)

        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients


    def plot_img_attributions(baseline,
                              image,
                              m_steps=100,
                              cmap=None,
                              overlay_alpha=0.6,
                              label=None,
                              targets=None):


        #base = baseline.numpy() - baseline.numpy().min()
        img = image.numpy() - image.numpy().min()
        img = img / img.max()
        
        if not targets:
            targets = labels
        
        fig, axs = plt.subplots(nrows=2, ncols=1+len(targets), squeeze=False, figsize=(4+4*len(targets), 8))

        #axs[0, 0].set_title('Baseline image')
        #axs[0, 0].imshow(base)
        axs[0, 0].axis('off')

        title = 'Original image' if label is None else 'Original image ({})'.format(label)
        axs[1, 0].set_title(title)
        axs[1, 0].imshow(img)
        axs[1, 0].axis('off')
        
        
        
        for i, l in enumerate(targets):
            attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=labels.index(l),
                                      m_steps=m_steps)
            attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1).numpy()
            #attribution_mask = tf.reduce_sum(attributions, axis=-1).numpy()
            
            # add blur for visibility
            attribution_mask = gaussian_filter(attribution_mask, sigma=2)
        
            axs[0, i+1].set_title('Attribution mask ({})'.format(l))
            mn = attribution_mask.min()
            mx = attribution_mask.max()
            m = axs[0, i+1].imshow(attribution_mask, cmap=cmap)#, norm=colors.SymLogNorm(vmin=mn, vmax=mx, linthresh=0.0001))
            plt.colorbar(m, ax=axs[0, i+1])
            axs[0, i+1].axis('off')

            axs[1, i+1].set_title('Overlay ({})'.format(l))
            axs[1, i+1].imshow(attribution_mask, cmap=cmap)
            axs[1, i+1].imshow(img, alpha=overlay_alpha)
            axs[1, i+1].axis('off')
            
        plt.tight_layout()
        return fig

    targets = [ "USH2A", "ABCA4" ]
    for label in targets:
        print("Generating images for " + label)
        elems = np.arange(len(x_test))[y_test[:, labels.index(label)] > 0]

        for i in np.random.choice(elems, 3):
            baseline = tf.Variable(np.mean(x_test, axis=0))
            input_image = tf.Variable(x_test[i])
            #label = labels[int(y_test[i].argmax())]

            plot_img_attributions(baseline, input_image, cmap="coolwarm", label=label, targets=targets).savefig(os.path.join("attributions", "example_{}.png".format(i)))
    