import argparse
import tensorflow as tf
import data_util
import FM_model
import numpy as np
import codecs


parser = argparse.ArgumentParser(description="""
FactorizationMachine is a library using Factorization Machine model 
to solve the problem about regression, classification and prediction.  
""")
# command line parameters
parser.add_argument("-b","--batch_size", type=int, default=16,
                    help="size of mini-batch")
parser.add_argument("-e", "--train_epoch", type=int, default=500,
                    help="times to train the model")
parser.add_argument("-r", "--learning_rate", type=int, default=0.001,
                    help="learning rate for the model")
parser.add_argument("-d","--train_data_path", type=str, default=None,
                    help="path to load input data")
parser.add_argument("-t","--test_data_path", type=str, default=None,
                    help="path to load test data")
parser.add_argument("-f","--factor_dim", type=int, default=8,
                    help="dimension of the feature vector")
parser.add_argument("-x", "--use_cross_entropy", action="store_true", default=False,
                    help="use cross entropy as loss, else MSE is used")
parser.add_argument("-o","--dump_factors_path", type=str, default=None,
                    help="path to dump the feature vectors")

args = parser.parse_args()

batch_size = args.batch_size

tf.logging.set_verbosity(tf.logging.INFO)  # Show log info of TensorFlow

if args.train_data_path:
    # Loading data from file
    train_data = data_util.Data(path=args.train_data_path, batch_size=args.batch_size)
    if train_data.load_data():
        tf.logging.info("Train_data set loaded")
        args.feature_size = train_data.get_feature_size()
        data_size = train_data.get_data_size()
        epoch = args.train_epoch * data_size // batch_size  # Get the epoch number for the batch
        epoch = epoch if epoch > 0 else 1

        run_test = False
        if args.test_data_path:
            test_data = data_util.Data(path=args.test_data_path, batch_size=args.batch_size)
            run_test = test_data.load_data()

        model = FM_model.Model(args)
        model.build_model()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        n = 0
        avg_loss = 0
        for step in range(epoch):
            batch_x, batch_y = train_data.get_next_batch()  # Get every batch from data
            feed_dict = {model.x: batch_x, model.y: batch_y}

            _, loss = sess.run([model.get_optimizer(), model.get_loss_var()], feed_dict=feed_dict)
            losses.append(loss)                     # Store loss of every step
            if step * batch_size // data_size > n:       # print the loss when all data is trained for 1 time
                n = step * batch_size // data_size
                avg_loss = np.mean(losses)
                losses.clear()

                if run_test:
                    test_losses = []
                    test_epoch = (test_data.get_data_size()-1) // batch_size + 1
                    for _ in range(test_epoch):
                        test_x, test_y = test_data.get_next_batch()
                        feed_dict = {model.x: test_x, model.y: test_y}

                        test_loss = sess.run(model.get_loss_var(), feed_dict=feed_dict)
                        test_losses.append(test_loss)
                    avg_test_loss = np.mean(test_losses)
                    tf.logging.info("Epoch: " + str(n) + ", Average loss: " + str(avg_loss)+" Test loss: "+str(avg_test_loss))
                else:
                    tf.logging.info("Epoch: " + str(n) + ", Average loss: " + str(avg_loss))

        if len(losses) > 0:
            final_loss = np.mean(losses)
        else:
            final_loss = avg_loss
        tf.logging.info("Train finished! Final loss: "+str(final_loss))

        if run_test:
            test_losses = []
            test_epoch = (test_data.get_data_size() - 1) // batch_size + 1
            for _ in range(test_epoch):
                test_x, test_y = test_data.get_next_batch()
                feed_dict = {model.x: test_x, model.y: test_y}

                test_loss = sess.run(model.get_loss_var(), feed_dict=feed_dict)
                test_losses.append(test_loss)
            avg_test_loss = np.mean(test_losses)
            tf.logging.info("Final test loss: " + str(avg_test_loss))

        if args.dump_factors_path:
            with codecs.open(args.dump_factors_path, "w", "utf8") as file:
                v = model.get_v(sess)
                feature_map = train_data.get_feature_map()
                for i in range(args.feature_size):
                    file.write(feature_map[i])
                    file.write(" ")
                    file.write(str(list(v[i])))
                    file.write("\n")
                tf.logging.info("Factor vectors dumped to "+args.dump_factors_path)

