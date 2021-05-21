import os
import sys
import argparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# from tensorflow.python.framework import ops
from dataset import Dataset
from model import Model
import prune
import train
import test


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--datasource', type=str, default='cifar-10', help='dataset to use')#mnist
    parser.add_argument('--path_data', type=str, default='./data', help='location to dataset')
    parser.add_argument('--aug_kinds', nargs='+', type=str, default=['fliplr','translate_px'], help='augmentations to perform')
    # Model options
    parser.add_argument('--arch', type=str, default='vgg-d', help='network architecture to use')#lenet5
    parser.add_argument('--target_sparsity', type=float, default=0.95, help='level of sparsity to achieve')
    # Train options
    parser.add_argument('--batch_size', type=int, default=128, help='number of examples per mini-batch')
    parser.add_argument('--train_iterations', type=int, default=200, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='momentum', help='optimizer of choice')#sgd
    parser.add_argument('--lr_decay_type', type=str, default='piecewise', help='learning rate decay type')#constant
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_boundaries', nargs='+', type=int, default=[30000 ,60000 ,90000 ,120000], help='boundaries for piecewise_constant decay')
    parser.add_argument('--decay_values', nargs='+', type=float, default=[0.1 ,0.02 ,0.004 ,0.0008 ,0.00016], help='values for piecewise_constant decay')
    # Initialization
    parser.add_argument('--initializer_w_bp', type=str, default='vs', help='initializer for w before pruning')
    parser.add_argument('--initializer_b_bp', type=str, default='zeros', help='initializer for b before pruning')
    parser.add_argument('--initializer_w_ap', type=str, default='vs', help='initializer for w after pruning')
    parser.add_argument('--initializer_b_ap', type=str, default='zeros', help='initializer for b after pruning')
    # Logging, saving, options
    parser.add_argument('--logdir', type=str, default='./reproduce-vgg', help='location for summaries and checkpoints')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    args = parser.parse_args()
    # Add more to args
    args.path_summary = os.path.join(args.logdir, 'summary')
    args.path_model = os.path.join(args.logdir, 'model')
    args.path_assess = os.path.join(args.logdir, 'assess')
    return args


def main():
    args = parse_arguments()

    # Dataset
    dataset = Dataset(**vars(args))

    # Reset the default graph and set a graph-level seed
    tf.reset_default_graph()
    # ops.reset_default_graph()
    tf.set_random_seed(9)
    # tf.random.set_seed(9)

    # Model
    model = Model(num_classes=dataset.num_classes, **vars(args))
    model.construct_model()

    # Session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # Prune
    prune.prune(args, model, sess, dataset)

    # Train and test
    train.train(args, model, sess, dataset)
    test.test(args, model, sess, dataset)

    sess.close()
    sys.exit()


if __name__ == "__main__":
    main()
