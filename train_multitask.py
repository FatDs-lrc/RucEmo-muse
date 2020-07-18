import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data
from utils.tools import calc_total_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def test(model, tst_iter):
    pass

def eval(model, val_iter):
    model.eval()
    total_pred_a = []
    total_label_a = []
    total_pred_v = []
    total_label_v = []
    total_length = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        pred_a = remove_padding(model.output_a.detach().cpu().numpy(), lengths)
        label_a = remove_padding(data['arousal'].numpy(), lengths)
        pred_v = remove_padding(model.output_v.detach().cpu().numpy(), lengths)
        label_v = remove_padding(data['valence'].numpy(), lengths)
        total_pred_a += pred_a
        total_label_a += label_a
        total_pred_v += pred_v
        total_label_v += label_v
    
    # calculate metrics
    total_pred_a = scratch_data(total_pred_a)
    total_label_a = scratch_data(total_label_a)
    total_pred_v = scratch_data(total_pred_v)
    total_label_v = scratch_data(total_label_v)
    mse_a, rmse_a, pcc_a, ccc_a = evaluate_regression(total_label_a, total_pred_a)
    mse_v, rmse_v, pcc_v, ccc_v = evaluate_regression(total_label_v, total_pred_v)
    model.train()

    return ccc_a, ccc_v

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name)   # get logger path
    suffix = opt.name                                   # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    
    dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)                         # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
                                                        # calculate input dims
    input_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.feature_set.split(','))))
    setattr(opt, "input_dim", input_dim)                # set input_dim attribute to opt
    model = create_model(opt, logger=logger)    # create a model given opt.model and other options
    model.setup(opt)                            # regular setup: load and print networks; create schedulers
    total_iters = 0                             # the total number of training iterations
    best_eval_ccc = 0                           # record the best eval UAR
    best_eval_ccc_a = 0
    best_eval_ccc_v = 0
    best_eval_epoch = -1                        # record the best eval epoch

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
    
        # eval val set
        ccc_a, ccc_v = eval(model, val_dataset)
        ccc_total = ccc_a + ccc_v
        logger.info('Val result of epoch %d / %d Arousal ccc: %.4f Valence ccc: %.4f' % (epoch, opt.niter + opt.niter_decay, ccc_a, ccc_v))
        if ccc_total > best_eval_ccc:
            best_eval_epoch = epoch
            best_eval_ccc = ccc_total
            best_eval_ccc_a = ccc_a
            best_eval_ccc_v = ccc_v
    # print best eval result
    logger.info('Best eval epoch %d found with ccc %f' % (best_eval_epoch, best_eval_ccc))
    # write to result dir
    autorun_result_dir = 'autorun/results'
    f = open(os.path.join(autorun_result_dir, opt.name + '.txt'), 'w')
    f.write('Best eval epoch %d found with ccc Arousal:%f Valence:%f' % (best_eval_epoch, best_eval_ccc_a, best_eval_ccc_v))
    f.close()
