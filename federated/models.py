import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
from typing import Any, Tuple, List, Union

from common_utils.classes import (
    Decoder,
    Encoder,
    FullyConected,
    T_predictor,
    Y_predictor
)

from common_utils.supervised_metrics import PEHE, RPEHE

keras = tf.keras


class LocalTEDVAE(keras.Model):
    def __init__(self, latent_dim_t: int, latent_dim_c: int, latent_dim_y: int, data_types: List[str], num_layers: int, num_neurons: int,
    loss_weights: Union[dict, pd.DataFrame], *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim_t = latent_dim_t
        self.latent_dim_c = latent_dim_c
        self.latent_dim_y = latent_dim_y
        self.num_layers = num_layers
        self.encoder_t = Encoder(latent_dim_t, num_layers, num_neurons)  # in tedvae paper, t does not have hidden layers
        self.encoder_c = Encoder(latent_dim_c, num_layers, num_neurons)
        self.encoder_y = Encoder(latent_dim_y, num_layers, num_neurons)
        self.decoder = Decoder(data_types, num_layers, num_neurons, binary_approach="bernoulli", temperature=.7)
        self.t_nn_model = T_predictor()
        self.t_nn_guide = T_predictor()

        self.y0_nn_model = Y_predictor(num_layers, num_neurons)
        self.y1_nn_model = Y_predictor(num_layers, num_neurons)

        self.y_nn_guide = FullyConected(num_layers=num_layers - 1, neurons=num_neurons, out_dim=num_neurons)
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

        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.loss_weights = {
                'w_disentanglement': 100,
                'w_kl': 1,
                'w_reconstruction': 1,
                'w_model':1
            }

        # Metrics
        self.loss_reconstruction = keras.metrics.Mean()
        self.loss_disentanglement = keras.metrics.Mean()
        self.loss_q_y = keras.metrics.Mean()
        self.loss_q_t = keras.metrics.Mean()
        self.elbo = keras.metrics.Mean()
        self.loss_kl = keras.metrics.Mean()
        self.loss_p_y = keras.metrics.Mean()
        self.loss_p_t = keras.metrics.Mean()

    def call(self, inputs, training=None, mask=None):

        x = inputs

        # ENCODER (GUIDE)
        q_zt = self.encoder_t(x)
        zt_samples = q_zt.sample()

        q_zc = self.encoder_c(x)
        zc_samples = q_zc.sample()

        q_zy = self.encoder_y(x)
        zy_samples = q_zy.sample()

        z_samples = tf.concat((zt_samples, zc_samples, zy_samples), axis=1)
        z_tc = tf.concat((zt_samples, zc_samples), axis=1)
        z_cy = tf.concat((zc_samples, zy_samples), axis=1)

        # DECODER (MODEL)
        p_x = self.decoder(z_samples, mode='exact')  # self.decoder.predict(z_samples)
        x_rec = p_x.sample()

        # MODEL
        p_t = self.t_nn_model(z_tc)
        t_model = p_t.sample()

        params0_model = self.y0_nn_model(z_cy)
        params1_model = self.y1_nn_model(z_cy)
        p_y0 = self.y0_nn_model.obs_model(*params0_model)
        p_y1 = self.y0_nn_model.obs_model(*params1_model)

        y0_model = p_y0.submodules[0].loc  # p_y0_model.loc
        y1_model = p_y1.submodules[0].loc  # p_y1_model.loc

        # GUIDE
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

    def do_not_train_vae(self, x_real, t_real, y_real):

        # print('do not train vae')

        out = self(x_real)

        loss_reconstruction = self.loss_weights['w_reconstruction'] * (-tf.reduce_mean(out['p_x'].log_prob(x_real)))

        # TEDVAE LOSSES
        # KL DIVERGENCE
        # substitute kl divergence by its terms
        loss_p_zt = self.prior_zt.log_prob(out['zt'])
        loss_p_zc = self.prior_zc.log_prob(out['zc'])
        loss_p_zy = self.prior_zy.log_prob(out['zy'])

        loss_q_zt = out['q_zt'].log_prob(out['zt'])
        loss_q_zc = out['q_zc'].log_prob(out['zc'])
        loss_q_zy = out['q_zy'].log_prob(out['zy'])

        loss_kl_zt = tf.reduce_mean(loss_q_zt - loss_p_zt)
        loss_kl_zc = tf.reduce_mean(loss_q_zc - loss_p_zc)
        loss_kl_zy = tf.reduce_mean(loss_q_zy - loss_p_zy)

        loss_kl = self.loss_weights['w_kl'] * (loss_kl_zt + loss_kl_zc + loss_kl_zy)

        t_real_bool = tf.expand_dims(tf.cast(t_real, 'bool'), 1)
        t_real = tf.expand_dims(t_real, 1)
        y_real = tf.expand_dims(y_real, 1)

        # MODEL
        params_model = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_model'], out['params1_model'])]

        p_y_given_z_post = self.y0_nn_model.obs_model(*params_model)

        t_samples_model = tf.cast(out['t_model'], 'float32')

        loss_p_t = -tf.reduce_mean(out['p_t'].log_prob(t_real))
        loss_p_y = - tf.reduce_mean(p_y_given_z_post.log_prob(y_real))

        loss_model = self.loss_weights['w_model'] * (loss_p_t + loss_p_y)

        # GUIDE/ INFERENCE

        params_guide = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_guide'], out['params1_guide'])]

        q_y_given_z_post = self.y0_nn_guide.obs_model(*params_guide)

        t_samples_guide = tf.cast(out['t_guide'], 'float32')

        loss_q_t = -tf.reduce_mean(out['q_t'].log_prob(t_real))
        loss_q_y = - tf.reduce_mean(q_y_given_z_post.log_prob(y_real))

        # losses of predictors for disentanglement
        loss_disentanglement = self.loss_weights['w_disentanglement'] * (loss_q_t + loss_q_y)


        return {
            "loss_reconstruction": loss_reconstruction,
            "loss_kl": loss_kl,
            "loss_disentanglement": loss_disentanglement,
            "loss_q_y": loss_q_y * self.loss_weights['w_disentanglement'],
            "loss_q_t": loss_q_t * self.loss_weights['w_disentanglement'],
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
    def train_vae(self, x_real, t_real, y_real):

        # print('training vae')

        with tf.GradientTape(persistent=False) as tape:
            out = self(x_real)

            loss_reconstruction = self.loss_weights['w_reconstruction']*(-tf.reduce_mean(out['p_x'].log_prob(x_real)))

            # TEDVAE LOSSES
            # KL DIVERGENCE
            # substitute kl divergence by its terms
            loss_p_zt = self.prior_zt.log_prob(out['zt'])
            loss_p_zc = self.prior_zc.log_prob(out['zc'])
            loss_p_zy = self.prior_zy.log_prob(out['zy'])

            loss_q_zt = out['q_zt'].log_prob(out['zt'])
            loss_q_zc = out['q_zc'].log_prob(out['zc'])
            loss_q_zy = out['q_zy'].log_prob(out['zy'])

            loss_kl_zt = tf.reduce_mean(loss_q_zt - loss_p_zt)
            loss_kl_zc = tf.reduce_mean(loss_q_zc - loss_p_zc)
            loss_kl_zy = tf.reduce_mean(loss_q_zy - loss_p_zy)

            loss_kl = self.loss_weights['w_kl']*(loss_kl_zt + loss_kl_zc + loss_kl_zy)

            t_real_bool = tf.expand_dims(tf.cast(t_real, 'bool'), 1)
            t_real = tf.expand_dims(t_real, 1)
            y_real = tf.expand_dims(y_real, 1)

            # MODEL
            params_model = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_model'], out['params1_model'])]

            p_y_given_z_post = self.y0_nn_model.obs_model(*params_model)

            t_samples_model = tf.cast(out['t_model'], 'float32')

            loss_p_t = -tf.reduce_mean(out['p_t'].log_prob(t_real))
            loss_p_y = - tf.reduce_mean(p_y_given_z_post.log_prob(y_real))

            loss_model = self.loss_weights['w_model']*(loss_p_t + loss_p_y)

            # GUIDE/ INFERENCE

            params_guide = [tf.where(t_real_bool, p1, p0) for p0, p1 in zip(out['params0_guide'], out['params1_guide'])]

            q_y_given_z_post = self.y0_nn_guide.obs_model(*params_guide)

            t_samples_guide = tf.cast(out['t_guide'], 'float32')

            loss_q_t = -tf.reduce_mean(out['q_t'].log_prob(t_real))
            loss_q_y = - tf.reduce_mean(q_y_given_z_post.log_prob(y_real))

            # losses of predictors for disentanglement
            loss_disentanglement = self.loss_weights['w_disentanglement'] * (loss_q_t + loss_q_y)

            loss = loss_reconstruction + loss_kl + loss_disentanglement + loss_model

        """
        train encoder + decoder with all losses
            - reconstruction to both
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
            "loss_q_y": loss_q_y * self.loss_weights['w_disentanglement'],
            "loss_q_t": loss_q_t * self.loss_weights['w_disentanglement'],
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

    def ite(self, x, y_scaler, predictor='model', num_samples=10, y_factual=None, t_factual=None):
        '''

           :param x: input data
           :param y_scaler: sklearn scaler fitted
           :param predictor: str, from which modules is predicted the outcome. In the original paper, it comes from the model
           :param num_samples: int, number of times the outcome is computed, to get expectation
           :return: the mean of the ites computed, the predicted y0 and the predicted y1 (arrays of 2D, n_patients x n_samples).
        '''
        y0_pred_list = []
        y1_pred_list = []
        ites_list = []
        for i in range(num_samples):
            out = self(x)

            y0_pred = out['y0_' + predictor]
            y1_pred = out['y1_' + predictor]

            if y_factual is not None and t_factual is not None:
                y0_pred = (1-t_factual)*y_factual + t_factual*y0_pred
                y1_pred = t_factual*y_factual + (1-t_factual)*y1_pred

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


class FedTEDVAE(keras.Model):
    def __init__(self, federated_strategy: str, num_domains: int, latent_dim_t: int, latent_dim_c: int, latent_dim_y: int, data_types: List[str],
                 num_layers: int,
                 num_neurons: int,
                 num_epochs_per_fed_average: int, loss_weights: dict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.federated_strategy = federated_strategy
        self.num_domains = num_domains
        self.latent_dim_t = latent_dim_t
        self.latent_dim_c = latent_dim_c
        self.latent_dim_y = latent_dim_y
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.loss_weights = loss_weights
        self.local_units = [LocalTEDVAE(latent_dim_t, latent_dim_c, latent_dim_y, data_types, num_layers, num_neurons, loss_weights) for _ in range(
            self.num_domains)]

        self.server = LocalTEDVAE(latent_dim_t, latent_dim_c, latent_dim_y, data_types, num_layers, num_neurons, loss_weights)

        self.current_epoch = tf.Variable(0.0, trainable=False)
        self.num_epochs_per_federate_average = tf.cast(tf.constant(num_epochs_per_fed_average), 'float32')
        self.do_federated_average = tf.Variable(True, trainable=False)

        # Metrics
        self.local_metrics = [keras.metrics.Mean() for _ in range(self.num_domains)]
        self.local_partial_metrics = [{
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
        } for _ in range(self.num_domains)]


    def call(self, inputs, training=None, mask=None):
        #
        out_list = []


        for i in range(self.num_domains):
            x = inputs[i]

            # ENCODER (GUIDE)
            q_zt = self.local_units[i].encoder_t(x)
            zt_samples = q_zt.sample()

            q_zc = self.local_units[i].encoder_c(x)
            zc_samples = q_zc.sample()

            q_zy = self.local_units[i].encoder_y(x)
            zy_samples = q_zy.sample()

            z_samples = tf.concat((zt_samples, zc_samples, zy_samples), axis=1)
            z_tc = tf.concat((zt_samples, zc_samples), axis=1)
            z_cy = tf.concat((zc_samples, zy_samples), axis=1)

            # DECODER (MODEL)
            p_x = self.local_units[i].decoder(z_samples, mode='exact')  # self.decoder.predict(z_samples)
            x_rec = p_x.sample()

            # MODEL
            p_t = self.local_units[i].t_nn_model(z_tc)
            t_model = p_t.sample()

            params0_model = self.local_units[i].y0_nn_model(z_cy)
            params1_model = self.local_units[i].y1_nn_model(z_cy)
            p_y0 = self.local_units[i].y0_nn_model.obs_model(*params0_model)
            p_y1 = self.local_units[i].y0_nn_model.obs_model(*params1_model)

            y0_model = p_y0.submodules[0].loc  # p_y0_model.loc
            y1_model = p_y1.submodules[0].loc  # p_y1_model.loc

            # GUIDE
            q_t = self.local_units[i].t_nn_guide(z_tc)
            t_guide = q_t.sample()

            hidden = self.local_units[i].y_nn_guide(z_cy)
            params0_guide = self.local_units[i].y0_nn_guide(hidden)
            params1_guide = self.local_units[i].y1_nn_guide(hidden)

            q_y0 = self.local_units[i].y0_nn_guide.obs_model(*params0_guide)
            q_y1 = self.local_units[i].y0_nn_guide.obs_model(*params1_guide)

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
            out_list.append(dist_and_samples)

        return out_list

    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None,
                steps_per_execution=None, jit_compile=None, **kwargs):
        for local_unit in self.local_units:
            local_unit.compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    def train_step(self, data):

        step_data = data[0]  # data is a tuple of one element

        # flag is the last column of each dataset, so we are going to filter it
        aux_train_data = []
        aux_val_data = []

        for ds in range(self.num_domains):
            flag_train = step_data[ds][:, -1]
            mask_train = tf.equal(flag_train, 1)
            aux_train_data.append(tf.boolean_mask(step_data[ds], mask_train)[:, :-1])

            val_ds = ds + self.num_domains
            flag_val = step_data[val_ds][:, -1]
            mask_val = tf.equal(flag_val, 1)
            aux_val_data.append(tf.boolean_mask(step_data[val_ds], mask_val)[:, :-1]) #-2 because the last row is mu0 (not seen in training)

        train_data = tuple(aux_train_data)
        val_data = tuple(aux_val_data)

        nodes_lengths = [tf.cast(tf.shape(dataset)[0], 'float32') for dataset in train_data]
        treated_lengths = [tf.reduce_sum(dataset[:,-2]) for dataset in train_data]
        input_shape = train_data[0].shape[1] - 2

        for i in range(self.num_domains):
            losses = dict()


            # DO FEDERATED AVERAGE (IN CENTRAL NODE)
            if 'fedavg' in self.federated_strategy:
                tf.cond(self.do_federated_average, true_fn=lambda: self.federated_average(nodes_lengths, treated_lengths, input_shape),
                        false_fn=lambda: None)


            if 'fedavg' in self.federated_strategy:
                losses_vae = tf.cond(self.do_federated_average,
                                  true_fn=lambda: self.local_units[i].do_not_train_vae(train_data[i][:, :-2], train_data[i][:, -2], train_data[i][:, -1]),
                                  false_fn=lambda: self.local_units[i].train_vae(train_data[i][:, :-2], train_data[i][:, -2], train_data[i][:, -1]))

                losses.update(losses_vae)
            else:
                losses.update(self.local_units[i].train_vae(train_data[i][:, :-2], train_data[i][:, -2], train_data[i][:, -1]))

            losses.update(self.local_units[i].losses_val(val_data[i][:, :-2], val_data[i][:, -2], val_data[i][:, -1]))

            domain_loss = 0.0
            for k, v in losses.items():
                if k == 'elbo' or k == 'loss_disentanglement' or k == 'loss_model':  # to avoid sum elements twice
                    domain_loss += v

            self.local_metrics[i].update_state(domain_loss)
            for key in self.local_partial_metrics[i]:
                if key in losses:
                    self.local_partial_metrics[i][key].update_state(losses[key])

        result = {
            f"Domain[{i}] - TOTAL loss": self.local_metrics[i].result() for i in range(self.num_domains)
        }

        result.update(
            {f"Domain[{i}] {key}": self.local_partial_metrics[i][key].result() for key in self.local_partial_metrics[i] for i in range(
                self.num_domains)}
        )

        return result

    def build_local_unit(self, i, input_shape):
        self.local_units[i].build(input_shape=(None, input_shape))

    def client_update(self, model, mean_client_weights):
        # model_weights = model.trainable_variables
        # Assign the mean client weights to the server model.
        tf.nest.map_structure(lambda x, y: x.assign(y),
                              model.trainable_variables, mean_client_weights)
    def federated_average(self, nodes_lengths, treated_lengths, input_shape):
        # print('fed avg')
        '''
        This function does federated average. Regressors fedavg is weighted by the number of trated/control patients in each node, as well as the
        rest of the modules (shared), are weighted by the proportion of the patients of each node divided by the total number of patients.

        :param nodes_lenghts: list of lengths of datasets in each node
        :param treated_lenghts: list with the number of treated individuals in each node
        :param input_shape: total number of patients
        :return: None, the weights of each layer is updated
        '''

        #first of all, compute non-treated_lenghts
        non_treated_lenghts = [nodes_lengths[j] - treated_lengths[j] for j in range(len(nodes_lengths))]

        client_reg_weights = [] # weights of regressors
        client_enc_weights = [] # weights of encoder
        client_dec_weights = [] # weights of decoder
        reg_names = ['t_predictor', 'y_predictor', 'fully_conected']  # fully conected comes from tarnet structure of the guide.


        for i in range(self.num_domains):

            if not self.local_units[i].built:
                self.build_local_unit(i, input_shape)


            regressors_weights = []
            enc_weights = []
            dec_weights = []
            for layer in self.local_units[i].layers:
                for reg_name in reg_names:
                    if reg_name in layer.name:
                        regressors_weights.append(tf.nest.map_structure(tf.identity, layer.trainable_variables))

                if 'encoder' in layer.name:
                    enc_weights.append(tf.nest.map_structure(tf.identity, layer.trainable_variables))
                elif 'decoder' in layer.name:
                    dec_weights.append(tf.nest.map_structure(tf.identity, layer.trainable_variables))



            client_reg_weights.append(regressors_weights)
            client_enc_weights.append(enc_weights)
            client_dec_weights.append(dec_weights)

        # Calculate the weighted average of the client weights
        reg_avg_weights = []
        enc_avg_weights = []
        dec_avg_weights = []

        if 'vanilla' in self.federated_strategy: #vanilla federated learning: without propensity adaptation
            if 'reg' in self.federated_strategy or 'all' in self.federated_strategy:
                for i in range(len(client_reg_weights[0])):
                    module_reg_avg_weights = [] #this list iterates over each neurons layer within each module
                    for j in range(len(client_reg_weights[0][i])):
                        weighted_sum = tf.zeros_like(client_reg_weights[0][i][j], dtype='float32')
                        total_samples = 0
                        #FED AVG IN Y0: ONLY UPDATED WITH UNTREATED GROUP
                        if i ==2 or i==5: #i==2 and i==5 correspond to y0 predictor
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * nodes_lengths[k]
                                total_samples += nodes_lengths[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                        #FED AVG IN Y1: TREATED GROUP
                        elif i==3 or i==6:
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * nodes_lengths[k]
                                total_samples += nodes_lengths[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                        #FED AVG IN T: ALL GROUPS (FULLY CONNECTED)
                        else:
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * nodes_lengths[k]
                                total_samples += nodes_lengths[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                    reg_avg_weights.append(module_reg_avg_weights)
        else:
            if 'reg' in self.federated_strategy or 'all' in self.federated_strategy:
                for i in range(len(client_reg_weights[0])):
                    module_reg_avg_weights = []  # this list iterates over each neurons layer within each module
                    for j in range(len(client_reg_weights[0][i])):
                        weighted_sum = tf.zeros_like(client_reg_weights[0][i][j], dtype='float32')
                        total_samples = 0
                        # FED AVG IN Y0: ONLY UPDATED WITH UNTREATED GROUP
                        if i == 2 or i == 5:  # i==2 and i==5 correspond to y0 predictor
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * non_treated_lenghts[k]
                                total_samples += non_treated_lenghts[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                        # FED AVG IN Y1: TREATED GROUP
                        elif i == 3 or i == 6:
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * treated_lengths[k]
                                total_samples += treated_lengths[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                        # FED AVG IN T: ALL GROUPS (FULLY CONNECTED)
                        else:
                            for k in range(len(client_reg_weights)):
                                weighted_sum += client_reg_weights[k][i][j] * nodes_lengths[k]
                                total_samples += nodes_lengths[k]
                            module_reg_avg_weights.append(weighted_sum / total_samples)
                    reg_avg_weights.append(module_reg_avg_weights)



        if 'enc' in self.federated_strategy or 'all' in self.federated_strategy:
            for i in range(len(client_enc_weights[0])):
                module_enc_avg_weights = []  # this list iterates over each neurons layer within each module
                for j in range(len(client_enc_weights[0][i])):
                    weighted_sum = tf.zeros_like(client_enc_weights[0][i][j], dtype='float32')
                    total_samples = 0
                    for k in range(len(client_enc_weights)):
                        weighted_sum += client_enc_weights[k][i][j] * nodes_lengths[k]
                        total_samples += nodes_lengths[k]
                    module_enc_avg_weights.append(weighted_sum / total_samples)
                enc_avg_weights.append(module_enc_avg_weights)

        if 'dec' in self.federated_strategy or 'all' in self.federated_strategy:
            for i in range(len(client_dec_weights[0])):
                module_dec_avg_weights = []
                for j in range(len(client_dec_weights[0][i])):
                    weighted_sum = tf.zeros_like(client_dec_weights[0][i][j], dtype='float32')
                    total_samples = 0
                    for k in range(len(client_dec_weights)):
                        weighted_sum += client_dec_weights[k][i][j] * nodes_lengths[k]
                        total_samples += nodes_lengths[k]
                    module_dec_avg_weights.append(weighted_sum / total_samples)
                dec_avg_weights.append(module_dec_avg_weights)

        for i in range(len(self.local_units)):
            if 'reg' in self.federated_strategy or 'all' in self.federated_strategy:
                reg_num=0
                for layer_i in range(len(self.local_units[i].layers)):
                    for reg_name in reg_names:
                        if  reg_name in self.local_units[i].layers[layer_i].name:
                            self.client_update(self.local_units[i].layers[layer_i], reg_avg_weights[reg_num])
                            reg_num += 1
                            break
            if 'enc' in self.federated_strategy or 'all' in self.federated_strategy:
                enc_num = 0
                for layer_i in range(len(self.local_units[i].layers)):
                    if 'encoder' in self.local_units[i].layers[layer_i].name:
                        self.client_update(self.local_units[i].layers[layer_i], enc_avg_weights[enc_num])
                        enc_num += 1
                        break
            if 'dec' in self.federated_strategy or 'all' in self.federated_strategy:
                dec_num = 0
                for layer_i in range(len(self.local_units[i].layers)):
                    if 'decoder' in self.local_units[i].layers[layer_i].name:
                        self.client_update(self.local_units[i].layers[layer_i], dec_avg_weights[dec_num])
                        dec_num += 1
                        break
    def compute_ites(self, features, y_scaler,  y_factual=None, t_factual=None):
        ites_list = []
        for i in range(self.num_domains):
            y_i = y_factual[i] if y_factual is not None else None #todo: check dimensions
            t_i = t_factual[i] if t_factual is not None else None #todo: check dimensions
            ites, _, _ = self.local_units[i].ite(features[i], y_scaler, y_factual= y_i, t_factual=t_i)
            ites_list.append(ites)

        return ites_list

    def validate_test(self, datasets, ite_real_list, y_scaler):

        ites_test_list = self.compute_ites(datasets, y_scaler)
        pehe = []
        for i in range(self.num_domains):
            pehe.append(RPEHE()(ite_real_list[i], ites_test_list[i]))

        return pehe

    def validate_train(self, datasets, ite_real_list, y_scaler, y_factual, t_factual):
        ites_test_list = self.compute_ites(datasets, y_scaler, y_factual, t_factual)
        pehe = []
        for i in range(self.num_domains):
            pehe.append(RPEHE()(ite_real_list[i], ites_test_list[i]))

        return pehe


