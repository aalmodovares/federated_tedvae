import tensorflow as tf

class PrintOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_freq, print_0, epochs, print_last):
        super(PrintOutputCallback, self).__init__()
        self.print_freq = print_freq
        self.print_0 = print_0
        self.epochs=epochs
        self.print_last = print_last

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_freq == 0:
            print_bool = True
            if epoch == 0 and not self.print_0:
                print_bool = False
        elif self.print_last and (epoch+1)%self.epochs==0:
            print_bool=True
        else:
            print_bool = False


        if print_bool:
            print('\n Epoch', epoch)
            for i, log in enumerate(logs):
                if i % 2 == 0:
                    print(log + ': ' + str(round(logs[log], 2)), end=' - ')
                else:
                    print(log + ': ' + str(round(logs[log], 2)))

class FederatedAverageCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.do_federated_average.assign(epoch%self.model.num_epochs_per_federate_average==0)