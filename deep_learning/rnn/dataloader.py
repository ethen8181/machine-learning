import numpy as np
from keras.utils import to_categorical


__all__ = ['DataLoader']


class DataLoader:
    """Container for a dataset."""

    def __init__(self, images, labels, num_classes):
        if images.shape[0] != labels.shape[0]:
            raise ValueError('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self.num_classes = num_classes
        self._images = images
        self._labels = labels

        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle = True):
        """Return the next `batch_size` examples from this data set."""

        # shuffle for the first epoch
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_images_and_labels()

        if start + batch_size > self._num_examples:
            # retrieve the rest of the examples that does not add up to a full batch size
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            rest_images = self._images[start:self._num_examples]
            rest_labels = self._labels[start:self._num_examples]
            if shuffle:
                self._shuffle_images_and_labels()

            # complete the batch size from the next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            new_images = self._images[start:end]
            new_labels = self._labels[start:end]
            images = np.concatenate((rest_images, new_images), axis = 0)
            labels = np.concatenate((rest_labels, new_labels), axis = 0)
            return images, to_categorical(labels, self.num_classes)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return (self._images[start:end],
                    to_categorical(self._labels[start:end], self.num_classes))

    def _shuffle_images_and_labels(self):
        permutated = np.arange(self._num_examples)
        np.random.shuffle(permutated)
        self._images[permutated]
        self._labels[permutated]
