from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from models.optim.swa import SWA
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

def create_train_loader(dataset, opt):
  loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=opt.batch_size, 
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=True
  )
  return loader

def create_optimizer(model, opt):
  if opt.weight_decay > 0:
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, 
                                  weight_decay=opt.weight_decay)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
  if opt.use_swa:
    if opt.swa_auto:
      optimizer = SWA(optimizer, 
                      swa_start=opt.swa_start, 
                      swa_freq=opt.swa_freq, 
                      swa_lr=opt.swa_lr)
    else:
      optimizer = SWA(optimizer)
  return optimizer


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = create_optimizer(model, opt)
  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  if opt.mixed_precision:
    from apex import amp
    model, optimizer = amp.initialize(
      model, optimizer, opt_level=opt.opt_level, 
      max_loss_scale=opt.max_loss_scale)
    print('Using amp with opt level %s...' % opt.opt_level)
  else:
    amp = None
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, amp, start_epoch = load_model(
      model, opt.load_model, optimizer, amp, 
      opt.resume, opt.lr, opt.lr_step)

  print('Setting up data...')
  val_dataset = Dataset(opt, 'val')
  val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.task == 'car_pose_6dof':
    # pass loaded 3D models for debug visualisations
    trainer.set_models(val_dataset.models)

  if opt.test:
    if opt.use_swa:
      optimizer.swap_swa_sgd()
      train_dataset = Dataset(opt, 'train')
      train_loader = create_train_loader(train_dataset, opt)
      trainer.bn_update(train_loader)
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_dataset = Dataset(opt, 'train')
  train_loader = create_train_loader(train_dataset, opt)

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    apply_swa = opt.use_swa and epoch * len(train_loader) > opt.swa_start
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer, amp)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer, amp)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer, amp)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)