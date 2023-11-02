import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np


from common_utils.classes import (
    Decoder,
    Encoder,
    FullyConected,
    T_predictor,
    Y_predictor
)

keras = tf.keras


class LocalTEDVAE(keras.Model):
    def __init__(self, latent_dim_t, latent_dim_c, latent_dim_y, data_types, num_layers, num_neurons, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim_t = latent_dim_t
        self.latent_dim_c = latent_dim_c
        self.latent_dim_y = latent_dim_y
        self.num_layers = num_layers
        self.encoder_t = Encoder(latent_dim_t, num_layers, num_neurons) #in tedvae paper, t does not have hidden layers
        self.encoder_c = Encoder(latent_dim_c,num_layers, num_neurons)
        self.encoder_y = Encoder(latent_dim_y,num_layers, num_neurons)
        self.decoder = Decoder(data_types, num_layers, num_neurons, binary_approach="bernoulli", temperature=.7)

        self.t_nn_model = T_predictor()
        self.t_nn_guide = T_predictor()

        self.y0_nn_model = Y_predictor(num_layers, num_neurons)
        self.y1_nn_model = Y_predictor(num_layers, num_neurons)

        self.y_nn_guide = FullyConected(num_layers=num_layers-1, neurons=num_neurons, out_dim=num_neurons)
        self.y0_nn_guide = Y_predictor(num_layers=0, neurons=None)
        self.y1_nn_guide = Y_predictor(num_layers=0, neurons=None)

        self.prior_zt = tfp.distributions.MultivariateNormalDiag(
            loc=tf.repeat(0., latent_dim_t),
            scale_diag=tf.repeat(1., latent_dim_t)
        )
        self.prior_zc = tfp.distributions.MultivariateNormalDiag(
            loc=tf.repeat(0., latent_dim_c),
            scale_diag=tf.repeat(1., latent_dim_c)
        )
        self.prior_zy = tfp.distributions.MultivariateNormalDiag(
            loc=tf.repeat(0., latent_dim_y),
            scale_diag=tf.repeat(1., latent_dim_y)
        )

        # Metrics
        self.loss_reconstruction = keras.metrics.Mean()
        self.loss_discriminator = keras.metrics.Mean()
        self.loss_disentanglement = keras.metrics.Mean()
        self.loss_q_y = keras.metrics.Mean()
        self.loss_q_t = keras.metrics.Mean()
        self.elbo = keras.metrics.Mean()
        self.loss_kl = keras.metrics.Mean()
        self.loss_p_y = keras.metrics.Mean()
        self.loss_p_t = keras.metrics.Mean()


    def call(self, inputs, training=None, mask=None):

        x = inputs

        #ENCODER (GUIDE)
        q_zt= self.encoder_t(x)
        zt_samples = q_zt.sample()

        q_zc = self.encoder_c(x)
        zc_samples = q_zc.sample()

        q_zy = self.encoder_y(x)
        zy_samples = q_zy.sample()

        z_samples = tf.concat((zt_samples, zc_samples, zy_samples), axis=1)
        z_tc = tf.concat((zt_samples, zc_samples), axis=1)
        z_cy = tf.concat((zc_samples, zy_samples), axis=1)

        #DECODER (MODEL)
        p_x = self.decoder(z_samples, mode='exact')#self.decoder.predict(z_samples)
        x_rec = p_x.sample()

        #MODEL
        p_t= self.t_nn_model(z_tc)
        t_model = p_t.sample()

        params0_model = self.y0_nn_model(z_cy)
        params1_model = self.y1_nn_model(z_cy)
        p_y0 = self.y0_nn_model.obs_model(*params0_model)
        p_y1 = self.y0_nn_model.obs_model(*params1_model)

        y0_model = p_y0.submodules[0].loc #p_y0_model.loc
        y1_model = p_y1.submodules[0].loc #p_y1_model.loc

        #GUIDE
        q_t = self.t_nn_guide(z_tc)
        t_guide = q_t.sample()

        hidden = self.y_nn_guide(z_cy)
        params0_guide = self.y0_nn_guide(hidden)
        params1_guide = self.y1_nn_guide(hidden)

        q_y0 = self.y0_nn_guide.obs_model(*params0_guide)
        q_y1 = self.y0_nn_guide.obs_model(*params1_guide)

        y0_guide = q_y0.submodules[0].loc  # p_y0_model.loc
        y1_guide = q_y1.submodules[0].loc  # p_y1_model.loc


        dist_and_samples = {'p_t': p_t, 't_model': t_model,
                            'p_y0': p_y0, 'y0_model': y0_model,
                            'p_y1': p_y1, 'y1_model': y1_model,
                            'params0_model': params0_model, 'params1_model': params1_model,
                            'p_x': p_x, 'x_rec': x_rec,
                            'q_t': q_t, 't_guide': t_guide,
                            'q_y0': q_y0, 'y0_guide': y0_guide,
                            'q_y1': q_y1, 'y1_guide': y1_guide,
                            'params0_guide': params0_guide, 'params1_guide': params1_guide,
                            'q_zt': q_zt, 'zt': zt_samples,
                            'q_zc': q_zc, 'zc': zc_samples,
                            'q_zy': q_zy, 'zy': zy_samples,
                            'z_tc': z_tc, 'z_cy': z_cy, 'z_samples': z_samples
                            }
        return dist_and_samples

    def losses_val(self, data, t_real, y_real):

        out = self(data)

        t_real_bool = tf.expand_dims(tf.cast(t_real, 'bool'), 1)
        t_real = tf.expand_dims(t_real, 1)
        y_real = tf.expand_dims(y_real, 1)
        # MODEL
        params_model = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_model'], out['params1_model'])]

        p_y_given_z_post = self.y0_nn_model.obs_model(*params_model)
        loss_p_t = -tf.reduce_mean(out['p_t'].log_prob(t_real))
        loss_p_y = - tf.reduce_mean(p_y_given_z_post.log_prob(y_real))

        return {"val_loss_p_t": loss_p_t,
                "val_loss_p_y": loss_p_y}
    # @tf.function
    def train_vae(self, x_real, t_real, y_real):
        with tf.GradientTape(persistent=False) as tape:
            out = self(x_real)

            loss_reconstruction = -tf.reduce_mean(out['p_x'].log_prob(x_real))

            # KL DIVERGENCE
            #substitute kl divergence by its terms

            loss_p_zt = self.prior_zt.log_prob(out['zt'])
            loss_p_zc = self.prior_zc.log_prob(out['zc'])
            loss_p_zy = self.prior_zy.log_prob(out['zy'])

            loss_q_zt = out['q_zt'].log_prob(out['zt'])
            loss_q_zc = out['q_zc'].log_prob(out['zc'])
            loss_q_zy = out['q_zy'].log_prob(out['zy'])

            loss_kl_zt = tf.reduce_mean(loss_q_zt - loss_p_zt)
            loss_kl_zc = tf.reduce_mean(loss_q_zc - loss_p_zc)
            loss_kl_zy = tf.reduce_mean(loss_q_zy - loss_p_zy)

            loss_kl = loss_kl_zt + loss_kl_zc + loss_kl_zy

            t_real_bool = tf.expand_dims(tf.cast(t_real, 'bool'), 1)
            t_real = tf.expand_dims(t_real, 1)
            y_real = tf.expand_dims(y_real, 1)

            # MODEL

            params_model = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_model'], out['params1_model'])]

            p_y_given_z_post = self.y0_nn_model.obs_model(*params_model)

            t_samples_model = tf.cast(out['t_model'], 'float32')

            loss_p_t = -tf.reduce_mean(out['p_t'].log_prob(t_real))
            loss_p_y = - tf.reduce_mean(p_y_given_z_post.log_prob(y_real))

            loss_model = loss_p_t + loss_p_y

            #GUIDE/ INFERENCE

            params_guide= [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_guide'], out['params1_guide'])]

            q_y_given_z_post = self.y0_nn_guide.obs_model(*params_guide)

            t_samples_guide = tf.cast(out['t_guide'], 'float32')

            loss_q_t = -tf.reduce_mean(out['q_t'].log_prob(t_real))
            loss_q_y = - tf.reduce_mean(q_y_given_z_post.log_prob(y_real))

            # losses of predictors for disentanglement
            loss_disentanglement = 100 * (loss_q_t + loss_q_y)

            loss = loss_reconstruction + loss_kl + loss_disentanglement + loss_model


        """
        train encoder + decoder with all losses
            - fool applies to decoder
            - reconstruction to both
            - and cycle to both as well
            - disentangled losses apply to encoder and regressors
        """

        trainable_variables = self.encoder_t.trainable_variables + self.encoder_c.trainable_variables + self.encoder_y.trainable_variables + \
                              self.decoder.trainable_variables + \
                              self.y0_nn_model.trainable_variables + self.y1_nn_model.trainable_variables + \
                              self.y_nn_guide.trainable_variables + self.y0_nn_guide.trainable_variables + self.y1_nn_guide.trainable_variables + \
                              self.t_nn_model.trainable_variables + self.t_nn_guide.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {
            "loss_reconstruction": loss_reconstruction,
            "loss_kl": loss_kl,
            "loss_disentanglement": loss_disentanglement,
            "loss_q_y": loss_q_y*100,
            "loss_q_t": loss_q_t*100,
            "elbo": loss_reconstruction + loss_kl,
            "loss_p_y": loss_p_y,
            "loss_p_t": loss_p_t,
            "loss_p_zt": loss_p_zt,
            "loss_p_zc": loss_p_zc,
            "loss_q_zt": loss_q_zt,
            "loss_q_zc": loss_q_zc,
            "loss_q_zy": loss_q_zy,
            "loss_model": loss_model
        }

class TEDVAE(keras.Model):
    def __init__(self, latent_dim_t, latent_dim_c, latent_dim_y, data_types, num_layers, num_neurons, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim_t = latent_dim_t
        self.latent_dim_c = latent_dim_c
        self.latent_dim_y = latent_dim_y
        self.num_layers = num_layers
        self.num_neurons  = num_neurons
        self.local_unit = LocalTEDVAE(latent_dim_t, latent_dim_c, latent_dim_y, data_types, num_layers, num_neurons)

        # Metrics
        self.local_metrics = keras.metrics.Mean()
        self.local_partial_metrics = {
            "loss_reconstruction": keras.metrics.Mean(),
            "loss_kl": keras.metrics.Mean(),
            "loss_disentanglement": keras.metrics.Mean(),
            "loss_q_y": keras.metrics.Mean(),
            "loss_q_t": keras.metrics.Mean(),
            "elbo": keras.metrics.Mean(),
            "loss_p_y": keras.metrics.Mean(),
            "loss_p_t": keras.metrics.Mean(),
            "loss_p_zt": keras.metrics.Mean(),
            "loss_p_zc": keras.metrics.Mean(),
            "loss_q_zt": keras.metrics.Mean(),
            "loss_q_zc": keras.metrics.Mean(),
            "loss_q_zy": keras.metrics.Mean(),
            "loss_model": keras.metrics.Mean(),
            "val_loss_p_t": keras.metrics.Mean(),
            "val_loss_p_y": keras.metrics.Mean()
        }

        # self.epoch_count = tf.Variable(0, dtype=tf.int32)

    def call(self, inputs, training=None, mask=None):
        out = self.local_unit(inputs)

        return out



    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None,
                steps_per_execution=None, jit_compile=None, **kwargs):
        local_unit = self.local_unit
        local_unit.compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    def train_step(self, data):

        losses = dict()
        _data = data[0]

        flag_train = _data[0][:, -1]
        mask_train = tf.equal(flag_train, 1)
        train_data = tf.boolean_mask(_data[0], mask_train, axis=0)[:,:-1]

        flag_val = _data[1][:, -1]
        mask_val = tf.equal(flag_val, 1)
        val_data = tf.boolean_mask(_data[1], mask_val, axis=0)[:, :-1]
        # val_data = val_data[:int(len(val_data)/2),:-1]


        losses.update(self.local_unit.train_vae(train_data[:, :-2], train_data[:, -2], train_data[:, -1]))
        losses.update(self.local_unit.losses_val(val_data[:, :-2], val_data[:, -2], val_data[:, -1]))

        domain_loss = 0.0
        for k, v in losses.items():
            if k=='elbo' or k=='loss_disentanglement' or k=='loss_model': #to avoid sum elements twice
                domain_loss += v

        self.local_metrics.update_state(domain_loss)
        for key in self.local_partial_metrics:
            self.local_partial_metrics[key].update_state(losses[key])


        result = {"TOTAL loss": self.local_metrics.result()}
        result.update(
            {f"{key}": self.local_partial_metrics[key].result() for key in self.local_partial_metrics}
        )
        return result


    def ite(self, x, y_scaler, predictor='model', num_samples=10, y_factual=None, t_factual=None):

        y0_pred_list = []
        y1_pred_list = []
        ites_list = []
        for i in range(num_samples):
            out = self(x)
            y0_pred = out['y0_' + predictor]
            y1_pred = out['y1_' + predictor]

            if y_factual is not None and t_factual is not None:
                y0_pred = (1 - t_factual) * y_factual + t_factual * y0_pred
                y1_pred = t_factual * y_factual + (1 - t_factual) * y1_pred

            y0_pred = y_scaler.inverse_transform(y0_pred)
            y1_pred = y_scaler.inverse_transform(y1_pred)

            y0_pred_list.append(y0_pred)
            y1_pred_list.append(y1_pred)

            ites = y1_pred - y0_pred

            ites_list.append(ites)

        y0_pred_arr = np.array(y0_pred_list).squeeze()
        y1_pred_arr = np.array(y1_pred_list).squeeze()
        ites_arr = y1_pred_arr - y0_pred_arr

        return ites_arr.mean(0), y0_pred_arr, y1_pred_arr

