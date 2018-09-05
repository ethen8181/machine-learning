import numpy as np
from scipy.ndimage import zoom
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array


class CAMExplainer:
    """CAM (Class Activation Map) Explainer"""

    def __init__(self, model, target_size=(224, 224)):
        self.model = model
        self.target_size = target_size

        self.class_weights_ = None
        self.resnet50_cam_layers_ = None

    def fit(self, img_path):
        if self.class_weights_ is None:
            self.resnet50_cam_layers_, self.class_weights_ = self._get_resnet50_cam_info()

        self.img_, keras_img = self._read_and_process_img(img_path)
        self.cam_, self.predicted_class_ = self._create_cam(keras_img)
        return self

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.imshow(self.img_, alpha=0.5)
        ax.imshow(self.cam_, cmap='jet', alpha=0.5)
        plt.show()

    def _create_cam(self, img):
        # before_gap_output will be of shape [1, 7, 7, 2048] for resnet50
        before_gap_output, prediction = self.resnet50_cam_layers_.predict(img)
        img_width = before_gap_output.shape[1]
        img_height = before_gap_output.shape[2]
        n_activation = before_gap_output.shape[3]

        predicted_class = np.argmax(prediction)
        dominate_class_weight = self.class_weights_[:, predicted_class]

        # we resize the shape of the activation so we can perform a dot product with
        # the dominated class weight
        before_gap_output = np.squeeze(before_gap_output).reshape((-1, n_activation))
        cam = np.dot(before_gap_output, dominate_class_weight).reshape((img_width, img_height))

        # we reshape it back to the target image size
        # so we can overlay the class activation map on top of our image later
        # order 1 = bi-linear interpolation was fast enough for this use-case
        width_scale_ratio = self.target_size[0] // img_width
        height_scale_ratio = self.target_size[1] // img_height
        cam = zoom(cam, (width_scale_ratio, height_scale_ratio), order=1)
        return cam, predicted_class

    def _read_and_process_img(self, img_path):
        """
        Reads in a single image, resize it to the specified target size
        and performs the same preprocessing on the image as the original
        pre-trained model.
        """
        img = load_img(img_path, target_size=self.target_size)
        img_arr = img_to_array(img)

        # keras works with batches of images, since we only have 1 image
        # here, we need to add an additional dimension to turn it into
        # shape [samples, size1, size2, channels]
        keras_img = np.expand_dims(img_arr, axis=0)

        # different pre-trained model preprocess the images differently
        # we also preprocess our images the same way to be consistent
        keras_img = preprocess_input(keras_img)
        return img, keras_img

    def _get_resnet50_cam_info(self):
        # we need the output of the activation layer right before the
        # global average pooling (gap) layer and the last dense/softmax
        # layer that generates the class prediction
        before_gap_layer = self.model.layers[-4]
        class_pred_layer = self.model.layers[-1]

        outputs = before_gap_layer.output, class_pred_layer.output
        resnet50_cam_layers = Model(inputs=self.model.input, outputs=outputs)

        # only access the first element of weights, we won't be needing the bias term here
        class_weights = class_pred_layer.get_weights()[0]
        return resnet50_cam_layers, class_weights
