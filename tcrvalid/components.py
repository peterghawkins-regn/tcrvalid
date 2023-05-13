# Copyright 2023 Regeneron Pharmaceuticals Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import mlflow
import os
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

def make_encoder(latent_dim):
    Seq_input = keras.Input(shape=(28, 8))

    # Block 1
    x = layers.Conv1D(32, kernel_size=5,strides=1,padding='same',kernel_initializer='he_uniform')(Seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    # Block 2
    x = layers.Conv1D(64, kernel_size=3,strides=1,padding='same',kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    # Block 3
    x = layers.Conv1D(128, kernel_size=3,strides=1,padding='same',kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    """
    # Block 4
    x = layers.Conv1D(256, kernel_size=3,strides=1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    """
    #x = layers.AveragePooling1D()(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(40,activation='relu')(x)
    encoded = x
    z_mean = layers.Dense(latent_dim,name="z_log_mean")(encoded)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoded)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(Seq_input, [z_mean, z_log_var, z], name="encoder")    
    # encoder.summary()
    return encoder

def make_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(28*128, activation='relu')(latent_inputs)
    x= layers.Reshape( (28,128) )(x)
    x = layers.Conv1DTranspose(128,kernel_size=3,padding='same',kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    """
    x = layers.Conv1DTranspose(128,kernel_size=3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    """
    x = layers.Conv1DTranspose(64,kernel_size=3,padding='same',kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1DTranspose(32,kernel_size=5,padding='same',kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    decoded_outputs = layers.Conv1DTranspose(8,kernel_size=1,padding='same')(x)
    decoder = keras.Model(latent_inputs, decoded_outputs, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, weight, capacity, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.insert_loss_tracker = keras.metrics.Mean(name="insert_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.weight = weight
        self.capacity = capacity
        #self.kl_loss_perDIM_tracker = tf.Variable(np.zeros((decoder.input.shape[1],)),dtype=tf.float32)
        #self.kl_loss_perDIM_tracker = []
     
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.insert_loss_tracker,
            #self.kl_loss_perDIM_tracker,
        ]
    
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            print('no')
            #sample_weight = None
            sample_weight = tf.ones_like(x,dtype=tf.float32)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x,training=True)
            #logvars = tf.reduce_mean(z_log_var,axis=0)
            #means = tf.reduce_mean(z_mean,axis=0)
            #mean_std = tf.math.reduce_std(z_mean,axis=0)
            
            reconstruction = self.decoder(z,training=True)
            print('recon')
            insert_loss = tf.reduce_mean(
                 tf.reduce_sum(
                     keras.losses.mean_squared_error(x, reconstruction)*sample_weight*x.shape[2],
                     #keras.losses.mean_squared_error(x, reconstruction)*x.shape[2],

                     axis=1
                 )
             )
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(x, reconstruction)*x.shape[2],
                    axis=1
                )
            )
            #print('recon_loss')
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #print("kl_loss = ",kl_loss)
            #print(K.int_shape(kl_loss)[1])
            
            kl_loss2=tf.reduce_sum(kl_loss, axis=1)
            
            kl_loss2 = tf.reduce_mean(kl_loss2)

            print("weight = ",self.weight)
            print("capacity = ",self.capacity)
            total_loss = reconstruction_loss + self.weight * K.abs(kl_loss2-self.capacity)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        #self.kl_loss_perDIM_tracker.update_state(kl_loss)
        
        self.total_loss_tracker.update_state(total_loss)
        self.insert_loss_tracker.update_state(insert_loss)

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss2)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "insert_loss": self.insert_loss_tracker.result(),
            #"kl_loss2": self.kl_loss_perDIM_tracker.result()

        }
    
    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = tf.ones_like(x,dtype=tf.float32)

        
        z_mean, z_log_var, z = self.encoder(x,training=False)

        reconstruction = self.decoder(z,training=False)
        insert_loss = tf.reduce_mean(
             tf.reduce_sum(
                 keras.losses.mean_squared_error(x, reconstruction)*sample_weight*x.shape[2],
                 #keras.losses.mean_squared_error(x, reconstruction)*x.shape[2],

                 axis=1
             )
         )
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(x, reconstruction)*x.shape[2],
                axis=1
            )
        )
        #reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(x, reconstruction)*sample_weight)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss2=tf.reduce_sum(kl_loss, axis=1)

        kl_loss2 = tf.reduce_mean(kl_loss2)

        total_loss = reconstruction_loss + self.weight * K.abs(kl_loss2-self.capacity)

        return {
            "loss": tf.reduce_mean(total_loss),
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss2),
            "insert_loss": tf.reduce_mean(insert_loss)
        }

class ComputeMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        logs['KL_loss_perDIM'] = self.model.kl_loss_perDIM.numpy() # Here I want to add the KL loss per DIM
        if (epoch + 1) % 10 == 0:

            logs['z_perDIM'] = np.nan # log of each latent space every 10 epochs need to parse tensor to numpy array
        else:
            logs['z_perDIM'] = np.nan
            
            
def save_loss_curve(hist):
    # makes a file on the driver node to save images to
    if not os.path.exists('images'):
        os.makedirs('images')
    f,ax = plt.subplots(1,1,figsize=(6,4))
    plt.plot(history.history['loss'],'k',label="Total loss")
    plt.plot(history.history['kl_loss'],'r',label="kl_loss")
    ax.plot(history.history['reconstruction_loss'],'b',label="recon loss")
    ax.plot(history.history['insert_loss'],'g',label="insert loss")
    ax.plot(history.history['val_loss'],'tab:orange', label="validation loss")
    plt.yscale('log')
    ax.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True,frameon=False)
    #bbox_to_anchor=(1.1, 1.05)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.tight_layout()
    plt.savefig('images/epoch_loss.png')
    return None

def collapse_metrics(limit):
    sdf=spark.read.format("parquet").load('/FileStore/peter.hawkins/tcr_qc_output/TrainTestSplit/tts_08_25_22/PQFormat/te')
    seqs = list(sdf.select('cdr2_cdr3').limit(20000).toPandas().cdr2_cdr3.values)
    x_train = mapping.seqs_to_array(seqs,maxlen=28)
    z_mean, z_log, z= encoder.predict(x_train[:20000,:,:])
    std_z_mu = z_mean.std(axis=0)
    logvars = z_log.mean(axis=0)
    variance = np.exp(logvars)
    #knn = NearestNeighbors(n_neighbors=2)
    #knn.fit(z_mean)
    #d_nn, _ = knn.kneighbors(z_mean, return_distance=True)
    #d_mean_mu = d_nn[:,1:].mean(axis=1).mean()
    #var_sqsum = np.sqrt(np.sum(var_df))
    #smooth_ratio = d_mean_mu/var_sqsum

    #d_vec = d_nn[:,1:].mean(axis=1)
    #fraction_inside_expectation = len(d_vec[d_vec<var_sqsum])/len(d_vec)
    #dropped_dims_fraction = len(logvars[logvars>-0.4])/len(logvars)
    #dropped_dims = len(logvars[logvars>-0.4])

    #means = z_mean.mean(axis=0)
    #ratio_metric=np.sqrt(np.exp(logvars))/mean_std
    ratio_metric=variance/std_z_mu
    ratio_metric_mean=ratio_metric.mean(axis=0)
    dataset = pd.DataFrame({'std_z_mu': std_z_mu, 'mean_logvar': logvars,'variance': variance,'ratio_metric': ratio_metric}, columns=['std_z_mu', 'mean_logvar','mean_vars','ratio_metric'])
    limit=2 # currently hardcoded but could be mades smarter variable
    collapsed = (dataset['ratio_metric'] > limit).sum()
    return dataset,collapsed,ratio_metric_mean


def make_vae(weight,capacity,latent_dim):
    #weight = K.variable(beta)
    #Capacity = K.variable(cap_dim*latent_dim)

    encoder = make_encoder(latent_dim)
    decoder = make_decoder(latent_dim)

    vae = VAE(encoder, decoder, weight, capacity*latent_dim)

    vae.compile(
        optimizer=keras.optimizers.Adam(),sample_weight_mode="temporal",
        metrics=[
          tf.keras.metrics.Accuracy()
        ]
    )
    return vae,encoder,decoder