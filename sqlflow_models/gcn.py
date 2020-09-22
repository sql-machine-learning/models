# Based on the code from: https://github.com/tkipf/keras-gcn
import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import scipy.sparse as sp
import numpy as np
import pickle, copy


class GCN(tf.keras.Model):
    def __init__(self, nhid, nclass, epochs, train_ratio, eval_ratio, 
                sparse_input=True, early_stopping=True, dropout=0.5, nlayer=2, feature_columns=None,
                id_col='id', feature_col='features', from_node_col='from_node_id', to_node_col='to_node_id'):
        """
        Implementation of GCN in this paper: https://arxiv.org/pdf/1609.02907.pdf. The original tensorflow implementation 
        is accessible here: https://github.com/tkipf/gcn, and one can find more information about GCN through: 
        http://tkipf.github.io/graph-convolutional-networks/.
        :param nhid: Number of hidden units for GCN.
            type nhid: int.
        :param nclass: Number of classes in total which will be the output dimension.
            type nclass: int.
        :param epochs: Number of epochs for the model to be trained.
            type epochs: int.
        :param train_ratio: Percentage of data points to be used for training.
            type train_ratio: float.
        :param eval_ratio: Percentage of data points to be used for evaluating.
            type eval_ratio: float.
        :param early_stopping: Whether to use early stopping trick during the training phase.
            type early_stopping: bool.
        :param dropout: The rate for dropout.
            type dropout: float.
        :param nlayer: Number of GCNLayer to be used in the model.
            type nlayer: int.
        :param feature_columns: a list of tf.feature_column. (Not used in this model)
            type feature_columns: list.
        :param id_col: Name for the column in database to be used as the id of each node.
            type id_col: string.
        :param feature_col: Name for the column in database to be used as the features of each node.
            type feature_col: string.
        :param from_node_col: Name for the column in database to be used as the from_node id of each edge.
            type from_node_col: string.
        :param to_node_col: Name for the column in database to be used as the to_node id of each edge.
            type to_node_col: string.
        """
        super(GCN, self).__init__()

        assert dropout < 1 and dropout > 0, "Please make sure dropout rate is a float between 0 and 1."
        assert train_ratio < 1 and train_ratio > 0, "Please make sure train_ratio is a float between 0 and 1."
        assert eval_ratio < 1 and eval_ratio > 0, "Please make sure eval_ratio is a float between 0 and 1."
        self.gc_layers = list()
        self.gc_layers.append(GCNLayer(nhid, kernel_regularizer=tf.keras.regularizers.l2(5e-4), sparse_input=sparse_input))
        for i in range(nlayer-1):
            self.gc_layers.append(GCNLayer(nhid, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
        self.gc_layers.append(GCNLayer(nclass))
        self.keep_prob = 1 - dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.nshape = None
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.nlayer = nlayer
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.sparse_input = sparse_input
        self.id_col = id_col
        self.feature_col = feature_col
        self.from_node_col = from_node_col
        self.to_node_col = to_node_col
        # try to load the result file
        try:
            with open('./results.pkl', 'rb') as f:
                self.results = pickle.load(f)
        except (FileNotFoundError, IOError):
            self.results = None

    def call(self, data):
        x, adj = data
        assert self.nshape is not None, "Should calculate the shape of input by preprocessing the data with model.preprocess(data)."
        if self.sparse_input:
            x = GCN.sparse_dropout(x, self.keep_prob, self.nshape)
        else:
            x = self.dropout(x)
        for i in range(self.nlayer-1):
            x = tf.keras.activations.relu(self.gc_layers[i](x, adj))
            x = self.dropout(x)
        x = self.gc_layers[-1](x, adj)

        return tf.keras.activations.softmax(x)

    def evaluate(self, data, y, sample_weight):
        """Function to evaluate the model."""
        return self.test(sample_weight, return_loss=True)

    def predict(self, data):
        """Function to predict labels with the model."""
        x, adj = data
        for i in range(self.nlayer-1):
            x = tf.keras.activations.relu(self.gc_layers[i](x, adj))
        x = self.gc_layers[-1](x, adj)
        return tf.keras.activations.softmax(x)

    @staticmethod
    def sparse_dropout(x, keep_prob, noise_shape):
        """Dropout for sparse tensors."""
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(x, dropout_mask)
        return pre_out * (1./keep_prob)

    @staticmethod
    def encode_onehot(labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    @staticmethod
    def normalize_adj(adjacency, symmetric=True):
        """
        Function to normalize the adjacency matrix (get the laplacian matrix).
        :param adjacency: Adjacency matrix of the dataset.
            type adjacency: Scipy COO_Matrix.
        :param symmetric: Boolean variable to determine whether to use symmetric laplacian.
            type symmetric: bool.
        """
        adjacency += sp.eye(adjacency.shape[0])
        if symmetric:
            """L=D^-0.5 * (A+I) * D^-0.5"""
            d = sp.diags(np.power(np.array(adjacency.sum(1)), -0.5).flatten(), 0)
            a_norm = adjacency.dot(d).transpose().dot(d).tocoo()
        else:
            """L=D^-1 * (A+I)"""
            d = sp.diags(np.power(np.array(adjacency.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adjacency).tocoo()

        return a_norm

    @staticmethod
    def normalize_feature(features, sparse_input):
        """Function to row-normalize the features input."""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        if sparse_input:
            return sp.csr_matrix(features).tocoo()
        else:
            return features

    def preprocess(self, ids, features, labels, edges):
        """Function to preprocess the node features and adjacency matrix."""
        if len(features.shape) > 2:
            features = np.squeeze(features)
        if len(edges.shape) > 2:
            edges = np.squeeze(edges)
        # sort the data in the correct order
        idx = np.argsort(np.array(ids))
        features = features[idx]
        labels = labels[idx]
        # preprocess
        features = GCN.normalize_feature(features, self.sparse_input)
        labels = GCN.encode_onehot(labels)
        adjacency = sp.coo_matrix((np.ones(len(edges)),
                    (edges[:, 0], edges[:, 1])),
                    shape=(features.shape[0], features.shape[0]), dtype="float32")

        adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(adjacency.T > adjacency)
        adjacency = GCN.normalize_adj(adjacency, symmetric=True)

        nf_shape = features.data.shape
        na_shape = adjacency.data.shape
        if self.sparse_input:
            features = tf.SparseTensor(
                        indices=np.array(list(zip(features.row, features.col)), dtype=np.int64),
                        values=tf.cast(features.data, tf.float32),
                        dense_shape=features.shape)
            features = tf.sparse.reorder(features)
        adjacency = tf.SparseTensor(
                        indices=np.array(list(zip(adjacency.row, adjacency.col)), dtype=np.int64),
                        values=tf.cast(adjacency.data, tf.float32),
                        dense_shape=adjacency.shape)
        adjacency = tf.sparse.reorder(adjacency)
        
        total_num = features.shape[0]
        train_num = round(total_num*self.train_ratio)
        eval_num = round(total_num*self.eval_ratio)
        train_index = np.arange(train_num)
        val_index = np.arange(train_num, train_num+eval_num)
        test_index = np.arange(train_num+eval_num, total_num)

        self.train_mask = np.zeros(total_num, dtype = np.bool)
        self.val_mask = np.zeros(total_num, dtype = np.bool)
        self.test_mask = np.zeros(total_num, dtype = np.bool)
        self.train_mask[train_index] = True
        self.val_mask[val_index] = True
        self.test_mask[test_index] = True

        print('Dataset has {} nodes, {} edges, {} features.'.format(features.shape[0], edges.shape[0], features.shape[1]))

        return features, labels, adjacency, nf_shape, na_shape
    
    def loss_func(self, model, x, y, train_mask, training=True):
        '''Customed loss function for the model.'''

        y_ = model(x, training=training)

        test_mask_logits = tf.gather_nd(y_, tf.where(train_mask))
        masked_labels = tf.gather_nd(y, tf.where(train_mask))

        return loss(labels=masked_labels, output=test_mask_logits)

    def grad(self, model, inputs, targets, train_mask):
        '''Calculate the gradients of the parameters.'''
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(model, inputs, targets, train_mask)
        
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    def test(self, mask, return_loss=False):
        '''Test the results on the model. Return accuracy'''
        logits = self.predict(data=[self.features, self.adjacency])

        test_mask_logits = tf.gather_nd(logits, tf.where(mask))
        masked_labels = tf.gather_nd(self.labels, tf.where(mask))

        ll = tf.equal(tf.argmax(masked_labels, -1), tf.argmax(test_mask_logits, -1))
        accuracy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

        if return_loss:
            loss_value = loss(labels=masked_labels, output=test_mask_logits)
            return [loss_value, accuracy]

        return accuracy

    def sqlflow_train_loop(self, x):
        """Customized training function."""
        # load data
        ids, ids_check, features, labels, edges, edge_check = list(), dict(), list(), list(), list(), dict()
        from_node = 0
        for inputs, label in x:
            id = inputs[self.id_col].numpy().astype(np.int32)
            feature = inputs[self.feature_col].numpy().astype(np.float32)
            from_node = inputs[self.from_node_col].numpy().astype(np.int32)
            to_node = inputs[self.to_node_col].numpy().astype(np.int32)
            if int(id) not in ids_check:
                ids.append(int(id))
                features.append(feature)
                labels.append(label.numpy()[0])
                ids_check[int(id)] = 0
            if tuple([int(from_node), int(to_node)]) not in edge_check:
                edge_check[tuple([int(from_node), int(to_node)])] = 0
                edges.append([from_node, to_node])
        features = np.stack(features)
        labels = np.stack(labels)
        edges = np.stack(edges)

        self.features, self.labels, self.adjacency, self.nshape, na_shape = self.preprocess(ids, features, labels, edges)
        # training the model
        wait = 0
        best_acc = -9999999
        PATIENCE = 10
        for epoch in range(self.epochs):
            # calculate the gradients and take the step
            loss_value, grads = self.grad(self, [self.features, self.adjacency], self.labels, self.train_mask)
            optimizer().apply_gradients(zip(grads, self.trainable_variables))
            # Test on train and evaluate dataset
            train_acc = self.test(self.train_mask)
            val_acc = self.test(self.val_mask)
            print("Epoch {} loss={:6f} accuracy={:6f} val_acc={:6f}".format(epoch, loss_value, train_acc, val_acc))
            # early stopping
            if epoch > 50 and self.early_stopping:
                if float(val_acc.numpy()) > best_acc:
                    best_acc = float(val_acc.numpy())
                    wait = 0
                else:
                    if wait >= PATIENCE:
                        print('Epoch {}: early stopping'.format(epoch))
                        break
                    wait += 1
        # evaluate the model
        result = self.evaluate(data=[self.features, self.adjacency], y=self.labels, sample_weight=self.val_mask)
        # get all the results
        predicted = self.predict([self.features, self.adjacency])
        # store the results in a pickled file
        with open('./results.pkl', 'wb') as f:
            results = dict()
            for i in range(len(ids)):
                results[str(ids[i])] = predicted[i]
            results['evaluation'] = result
            pickle.dump(results, f)
            self.results = results

    def sqlflow_evaluate_loop(self, x, metric_names):
        """Customed evaluation, can only support calculating the accuracy."""
        assert self.results is not None, "Please make sure to train the model first."
        eval_result = self.results['evaluation']
        return eval_result

    def sqlflow_predict_one(self, sample):
        """Customed prediction, sample must be the node id."""
        assert self.results is not None, "Please make sure to train the model first."
        prediction = self.results[str(int(sample))]
        return [prediction]

def optimizer():
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adam(lr=0.01)

def loss(labels, output):
    """Default loss function for classification task."""
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return criterion(y_true=labels, y_pred=output)

# Graph Convolutional Layer
class GCNLayer(tf.keras.layers.Layer):
 
    def __init__(self, units, use_bias=True, sparse_input=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """GCNLayer
        Graph Convolutional Networks Layer from paper: https://arxiv.org/pdf/1609.02907.pdf. This is used in the GCN model for 
        classification task on graph-structured data.
        :param units: Number of hidden units for the layer.
            type units: int.
        :param use_bias: Boolean variable to determine whether to use bias.
            type use_bias: bool.
        :param sparse_input: Boolean variable to check if input tensor is sparse.
            type sparse_input: bool.
        :param kernel_initializer: Weight initializer for the GCN kernel.
        :param bias_initializer: Weight initializer for the bias.
        :param kernel_regularizer: Weight regularizer for the GCN kernel.
        :param bias_regularizer: Weight regularizer for the bias.
        :param kernel_constraint: Weight value constraint for the GCN kernel.
        :param bias_constraint: Weight value constraint for the bias.
        :param kwargs:
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.sparse_input = sparse_input
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        self.built = True
    
    def call(self, inputs, adj, **kwargs):
        assert isinstance(adj, tf.SparseTensor), "Adjacency matrix should be a SparseTensor"
        if self.sparse_input:
            assert isinstance(inputs, tf.SparseTensor), "Input matrix should be a SparseTensor"
            support = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        else:
            support = tf.matmul(inputs, self.kernel)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        if self.use_bias:   
            output = output + self.bias
        else:
            output = output
        return output

    def get_config(self):
        config = {'units': self.units,
                  'use_bias': self.use_bias,
                  'sparse_input': self.sparse_input,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GCNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))