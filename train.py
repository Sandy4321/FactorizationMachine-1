import argparse
import tensorflow as tf
import data_util
import FM_model
import numpy as np
import codecs


parser = argparse.ArgumentParser(description="Parameters")
# command line parameters
parser.add_argument("--batch_size", type=int, default=16,
                    help="size of mini-batch")
parser.add_argument("--train_epoch", type=int, default=500,
                    help="times to train the model")
parser.add_argument("--learning_rate", type=int, default=0.001,
                    help="learning rate for the model")
parser.add_argument("--data_path", type=str, default=None,
                    help="path to load input data")
parser.add_argument("--factor_dim", type=int, default=8,
                    help="dimension of the feature vector")
parser.add_argument("--use_cross_entropy", type=bool, default=False,
                    help="use cross entropy as loss, else MSE is used")
parser.add_argument("--dump_factors_path", type=str, default=None,
                    help="path to dump the feature vectors")

args = parser.parse_args()

batch_size = args.batch_size

tf.logging.set_verbosity(tf.logging.INFO)  # Show log info of TensorFlow

if args.data_path:
    # Loading data from file
    data = data_util.Data(path=args.data_path, batch_size=args.batch_size)
    if data.load_data():
        tf.logging.info("Data set loaded")
        args.feature_size = data.get_feature_size()
        data_size = data.get_data_size()
        epoch = args.train_epoch * data_size // batch_size  # Get the epoch number for the batch
        epoch = epoch if epoch > 0 else 1

        model = FM_model.Model(args)
        model.build_model()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        n = 0
        avg_loss = 0
        for step in range(epoch):
            batch_x, batch_y = data.get_next_batch()  # Get every batch from data
            feed_dict = {model.x: batch_x, model.y: batch_y}

            _, loss = sess.run([model.get_optimizer(), model.get_loss_var()], feed_dict=feed_dict)
            losses.append(loss)                     # Store loss of every step
            if step * batch_size // data_size > n:       # print the loss when all data is trained for 1 time
                n = step * batch_size // data_size
                avg_loss = np.mean(losses)
                losses.clear()
                tf.logging.info("Epoch: "+str(n)+", Average loss: "+str(avg_loss))

        if len(losses) > 0:
            final_loss = np.mean(losses)
        else:
            final_loss = avg_loss
        tf.logging.info("Train finished! Final loss: "+str(final_loss))

        if args.dump_factors_path:
            with codecs.open(args.dump_factors_path, "w", "utf8") as file:
                v = model.get_v(sess)
                feature_map = data.get_feature_map()
                for i in range(args.feature_size):
                    file.write(feature_map[i])
                    file.write(" ")
                    file.write(str(list(v[i])))
                    file.write("\n")
                tf.logging.info("Factor vectors dumped to "+args.dump_factors_path)

