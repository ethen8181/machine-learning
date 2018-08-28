import numpy as np


class SoftmaxRegression:
    """
    Softmax regression classifier

    Parameters
    ------------
    eta : float
        learning rate, or so called step size (between 0.0 and 1.0)

    epochs : int
        number of passes over the training dataset (iterations),
        prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.

    minibatches : int
        The number of minibatches for gradient-based optimization.
        if len(y): gradient descent
        if 1: stochastic gradient descent (SGD) online learning
        if 1 < minibatches < len(y): SGD minibatch learning

    l2 : float, default 0
        l2 regularization parameter
        if 0: no regularization
    """

    def __init__(self, eta, epochs, minibatches, l2 = 0):
        self.eta = eta
        self.epochs = epochs
        self.minibatches = minibatches
        self.l2 = l2

    def fit(self, X, y):
        data_num = X.shape[0]
        feature_num = X.shape[1]
        class_num = np.unique(y).shape[0]

        # initialize the weights and bias
        self.w = np.random.normal(size = (feature_num, class_num))
        self.b = np.zeros(class_num)
        self.costs = []

        # one hot encode the output column and shuffle the data before starting
        y_encode = self._one_hot_encode(y, class_num)
        X, y_encode = self._shuffle(X, y_encode, data_num)

        # `i` keeps track of the starting index of
        # current batch, so we can do batch training
        i = 0

        # note that epochs refers to the number of passes over the
        # entire dataset, thus if we're using batches, we need to multiply it
        # with the number of iterations, we also make sure the batch size
        # doesn't exceed the number of training samples, if it does use batch size of 1
        iterations = self.epochs * max(data_num // self.minibatches, 1)

        for _ in range(iterations):
            batch = slice(i, i + self.minibatches)
            batch_X, batch_y_encode = X[batch], y_encode[batch]

            # forward and store the cross entropy cost
            net = self._net_input(batch_X)
            softm = self._softmax(net)
            error = softm - batch_y_encode
            cost = self._cross_entropy_cost(output = softm, y_target = batch_y_encode)
            self.costs.append(cost)

            # compute gradient and update the weight and bias
            gradient = np.dot(batch_X.T, error)
            self.w -= self.eta * (gradient + self.l2 * self.w)
            self.b -= self.eta * np.sum(error, axis = 0)

            # update starting index of for the batches
            # and if we made a complete pass over data, shuffle again
            # and refresh the index that keeps track of the batch
            i += self.minibatches
            if i + self.minibatches > data_num:
                X, y_encode = self._shuffle(X, y_encode, data_num)
                i = 0

        # stating that the model is fitted and
        # can be used for prediction
        self._is_fitted = True
        return self

    def _one_hot_encode(self, y, class_num):
        y_encode = np.zeros((y.shape[0], class_num))
        for idx, val in enumerate(y):
            y_encode[idx, val] = 1.0

        return y_encode

    def _shuffle(self, X, y_encode, data_num):
        permutation = np.random.permutation(data_num)
        X, y_encode = X[permutation], y_encode[permutation]
        return X, y_encode

    def _net_input(self, X):
        net = X.dot(self.w) + self.b
        return net

    def _softmax(self, z):
        softm = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
        return softm

    def _cross_entropy_cost(self, output, y_target):
        cross_entropy = np.mean(-np.sum(np.log(output) * y_target, axis = 1))
        l2_penalty = 0.5 * self.l2 * np.sum(self.w ** 2)
        cost = cross_entropy + l2_penalty
        return cost

    def predict_proba(self, X):
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet!')

        net = self._net_input(X)
        softm = self._softmax(net)
        return softm

    def predict(self, X):
        softm = self.predict_proba(X)
        class_labels = np.argmax(softm, axis = 1)
        return class_labels
