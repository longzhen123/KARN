from src.KARN import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=50, help='embedding size')
    # parser.add_argument('--n_path', type=int, default=6, help='the number of paths')
    # parser.add_argument('--n_record', type=int, default=10, help='the number of records')
    # parser.add_argument('--n_neighbor', type=int, default=20, help='the number of neighbors')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--n_path', type=int, default=6, help='the number of paths')
    # parser.add_argument('--n_record', type=int, default=10, help='the number of records')
    # parser.add_argument('--n_neighbor', type=int, default=20, help='the number of neighbors')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=10, help='embedding size')
    # parser.add_argument('--n_path', type=int, default=6, help='the number of paths')
    # parser.add_argument('--n_record', type=int, default=10, help='the number of records')
    # parser.add_argument('--n_neighbor', type=int, default=20, help='the number of neighbors')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=40, help='embedding size')
    parser.add_argument('--n_path', type=int, default=6, help='the number of paths')
    parser.add_argument('--n_record', type=int, default=10, help='the number of records')
    parser.add_argument('--n_neighbor', type=int, default=20, help='the number of neighbors')
    parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.901 	 train_acc: 0.829 	 eval_auc: 0.834 	 eval_acc: 0.771 	 test_auc: 0.832 	 test_acc: 0.773 		[0.16, 0.25, 0.4, 0.47, 0.47, 0.52, 0.54, 0.57]
book	train_auc: 0.746 	 train_acc: 0.671 	 eval_auc: 0.732 	 eval_acc: 0.685 	 test_auc: 0.728 	 test_acc: 0.685 		[0.11, 0.15, 0.31, 0.32, 0.32, 0.37, 0.41, 0.46]
ml	train_auc: 0.857 	 train_acc: 0.774 	 eval_auc: 0.850 	 eval_acc: 0.768 	 test_auc: 0.851 	 test_acc: 0.771 		[0.1, 0.21, 0.31, 0.33, 0.33, 0.38, 0.42, 0.45]
yelp    train_auc: 0.889 	 train_acc: 0.807 	 eval_auc: 0.852 	 eval_acc: 0.777 	 test_auc: 0.852 	 test_acc: 0.776 		[0.13, 0.21, 0.44, 0.45, 0.45, 0.48, 0.53, 0.55]

'''