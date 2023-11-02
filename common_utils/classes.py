import tensorflow_probability as tfp
import tensorflow as tf

keras = tf.keras

class GaussianObservationalStructure_y(keras.layers.Layer):
    def call(self, loc, scale, *args, **kwargs):

        out = tfp.distributions.Independent(
            tfp.distributions.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1
        )
        return out

class GaussianObservationalStructure(keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        loc, scale = tf.split(inputs, 2, axis=-1)
        loc = tf.clip_by_value(loc, -100, 100)
        scale = tf.math.softplus(tf.math.add(scale, 1e-3))
        scale = tf.clip_by_value(scale, 1e-3, 100)
        out = tfp.distributions.Independent(
            tfp.distributions.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1
        )
        return out


class BernoulliObservationalStructure(keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        logits = tf.clip_by_value(inputs, -10, 10)
        out = tfp.distributions.Independent(
            tfp.distributions.Bernoulli(logits=logits, validate_args=False),
            reinterpreted_batch_ndims=1
        )
        return out


class GumbelSoftmaxObservationalStructure(keras.layers.Layer):
    def __init__(self, temperature=.2, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.temperature = temperature

    def call(self, inputs, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode == "exact":
            x = tfp.distributions.Independent(
                tfp.distributions.Bernoulli(logits=inputs),
                reinterpreted_batch_ndims=1
            )
        else:
            x = tf.nn.tanh(inputs) * 2.
            x = tfp.distributions.Independent(
                tfp.distributions.RelaxedBernoulli(self.temperature, logits=x),
                reinterpreted_batch_ndims=1
            )
        return x


class Encoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, neurons,  **kwargs):
        super().__init__(**kwargs)
        self.dense_structure = keras.Sequential([
            keras.layers.Dense(neurons, activation="elu")#, kernel_initializer="glorot_normal")
            for _ in range(num_layers-1)] +
            [keras.layers.Dense(latent_dims * 2)]) # the output of the encoder includes scale and log of normal RV


    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        loc, scale = tf.split(x, 2, axis=-1)
        # scale = tf.exp(scale) + 1e-6
        scale = tf.nn.softplus(scale)
        out = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale, validate_args=False)
        return out

    def predict(self, inputs):
        out = self(inputs)
        return out


class Decoder(keras.layers.Layer):
    def __init__(self, data_types, num_layers, neurons, binary_approach="bernoulli", temperature=None, **kwargs):
        super().__init__(**kwargs)
        self.binary_approach = binary_approach
        self.data_types = data_types
        self.out_size = 0
        for elem in data_types:
            if elem == "binary":
                self.out_size += 1
            else:
                self.out_size += 2

        self.dense_structure = keras.Sequential([
            keras.layers.Dense(neurons, activation="relu", kernel_initializer="glorot_normal")
            for _ in range(num_layers)] +
            [keras.layers.Dense(self.out_size),
        ])

        self.obs_models = []
        for elem in data_types:
            if elem == "real":
                obs_model = GaussianObservationalStructure()
            elif elem == "binary" and self.binary_approach == "bernoulli":
                obs_model = BernoulliObservationalStructure()
            elif elem == "binary" and self.binary_approach == "gumbel-softmax":
                obs_model = GumbelSoftmaxObservationalStructure(temperature)
            self.obs_models.append(obs_model)

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        used = 0
        out = []
        for idx, elem in enumerate(self.data_types):
            if elem == "binary":
                x_sub = x[..., used:1 + used]
                used += 1
            else:
                x_sub = x[..., used:2 + used]
                used += 2

            out.append(
                self.obs_models[idx](x_sub, **kwargs)
            )
        return tfp.distributions.Blockwise(out, dtype_override=tf.float32)

    def predict(self, inputs):
        assert_op = tf.Assert(isinstance(inputs, tf.data.Dataset), ["Inputs is not a tf.data.Dataset type"])
        out = []
        with tf.control_dependencies([assert_op]):
            for step, data in enumerate(inputs):
                q_z_given_x = self(data)
                out.append(q_z_given_x)

        out = tf.concat(out, axis=0)
        return out


class T_predictor(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.out_size = 1
        self.dense_structure = keras.Sequential(
            [keras.layers.Dense(self.out_size, kernel_regularizer="l2"),
        ])
        self.obs_model = BernoulliObservationalStructure()

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        return self.obs_model(x)

class T_predictor_bce(keras.layers.Layer):
    def __init__(self, num_layers, neurons, **kwargs):
        super().__init__(**kwargs)

        self.out_size = 1
        self.dense_structure = keras.Sequential(
            [keras.layers.Dense(neurons, activation="elu", kernel_initializer="glorot_normal")
                for _ in range(num_layers)] +
            [keras.layers.Dense(self.out_size, activation='sigmoid'),
        ])

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        return x

    def predict(self, inputs):
        out = self(inputs)
        return out

class FullyConected(keras.layers.Layer):
    def __init__(self, num_layers, neurons, out_dim, **kwargs):
        super().__init__(**kwargs)


        self.dense_structure = keras.Sequential([
            keras.layers.Dense(neurons, activation="elu", kernel_regularizer="l2")
            for _ in range(num_layers)] + [keras.layers.Dense(out_dim, kernel_regularizer="l2"),])

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        return x


class Y_predictor(keras.layers.Layer):
    def __init__(self, num_layers, neurons, **kwargs):
        super().__init__(**kwargs)

        self.out_size = 2
        self.dense_structure = keras.Sequential([
            keras.layers.Dense(neurons, activation="elu", kernel_regularizer="l2")#, kernel_initializer="glorot_normal")
            for _ in range(num_layers)] +
        [keras.layers.Dense(self.out_size, kernel_regularizer="l2"),
        ])
        self.obs_model = GaussianObservationalStructure_y()

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        loc, scale = tf.split(x, 2, axis=-1)
        # loc = tf.nn.softplus(loc)
        loc = tf.clip_by_value(loc, -100, 100)
        scale = tf.math.softplus(tf.math.add(scale, 1e-3))
        scale = tf.clip_by_value(scale, 1e-3, 100)
        return loc, scale


class Y_predictor_mse(keras.layers.Layer):
    def __init__(self, num_layers, neurons, **kwargs):
        super().__init__(**kwargs)

        self.out_size = 1
        self.dense_structure = keras.Sequential([
            keras.layers.Dense(neurons, activation="elu", kernel_initializer="glorot_normal")
            for _ in range(num_layers)] +
        [keras.layers.Dense(self.out_size),
        ])

    # @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        x = self.dense_structure(inputs)
        return x


    def predict(self, inputs):
        out = self(inputs)
        return out