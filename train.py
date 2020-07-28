import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func
from utils.tools import calc_total_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def test(model, tst_iter):
    pass

def eval(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    total_length = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        pred = remove_padding(model.output.detach().cpu().numpy(), lengths)
        label = remove_padding(data[opt.target].numpy(), lengths)
        total_pred += pred
        total_label += label
    
    # calculate metrics
    best_window = None
    if smooth:
        total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=logger)

    total_pred = scratch_data(total_pred)
    total_label = scratch_data(total_label)
    mse, rmse, pcc, ccc = evaluate_regression(total_label, total_pred)
    model.train()

    return mse, rmse, pcc, ccc, best_window

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    smooth = False
    best_window = None
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name)   # get logger path
    suffix = opt.name                                   # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    
    dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)                         # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
                                                        # calculate input dims
    if opt.feature_set != 'None':
        input_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.feature_set.split(','))))
        setattr(opt, "input_dim", input_dim)                # set input_dim attribute to opt
    if hasattr(opt, "a_features"):
        a_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.a_features.split(','))))
        setattr(opt, "a_dim", a_dim)                # set a_dim attribute to opt
    if hasattr(opt, "v_features"):
        v_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.v_features.split(','))))
        setattr(opt, "v_dim", v_dim)                # set v_dim attribute to opt
    if hasattr(opt, "l_features"):
        l_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.l_features.split(','))))
        setattr(opt, "l_dim", l_dim)                # set l_dim attribute to opt
    
    model = create_model(opt, logger=logger)    # create a model given opt.model and other options
    model.setup(opt)                            # regular setup: load and print networks; create schedulers
    total_iters = 0                             # the total number of training iterations
    best_eval_ccc = 0                           # record the best eval UAR
    best_eval_epoch = -1                        # record the best eval epoch
    best_eval_window = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        iter_data_statis = 0.0          # record total data reading time
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            iter_data_statis += iter_start_time-iter_data_time
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.run()                     # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec, Data loading: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, iter_data_statis))
        model.update_learning_rate()                      # update learning rates at the end of every epoch.

        # eval trn set
        mse, rmse, pcc, ccc, window = eval(model, dataset)
        logger.info('Trn result of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))
        
        # eval val set
        mse, rmse, pcc, ccc, window = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))
        if ccc > best_eval_ccc:
            best_eval_epoch = epoch
            best_eval_ccc = ccc
            best_eval_window = window
    
    # print best eval result
    logger.info('Best eval epoch %d found with ccc %f' % (best_eval_epoch, best_eval_ccc))
    logger.info(opt.name)
    # record best window
    if smooth:
        f = open(os.path.join(opt.checkpoints_dir, opt.name, 'best_eval_window'), 'w')
        f.write(str(best_eval_window))
        f.close()
    
    # write to result dir
    clean_chekpoints(opt.name, best_eval_epoch)
    autorun_result_dir = 'autorun/results'
    f = open(os.path.join(autorun_result_dir, opt.name + '.txt'), 'w')
    f.write('Best eval epoch %d found with ccc %.4f' % (best_eval_epoch, best_eval_ccc))
    f.close()
