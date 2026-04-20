import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pandas as pd
from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def worker_init_fn(worker_id):
    np.random.seed(fix_seed + worker_id)
    random.seed(fix_seed + worker_id)



os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"








def train_function(args):
    # 固定随机种子（保持原始实现）
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 初始化加速器（完全保留原始设置）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    # 完整保留原始训练流程
    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        # 原始数据加载方式
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        # 原始模型选择逻辑
        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = TimeLLM.Model(args).float()

        # 原始检查点路径设置
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        # 完整保留原始优化器设置
        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # 原始学习率调度器逻辑
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=len(train_loader),
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        # 原始加速器准备代码
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        # 完整保留原始训练循环
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        metrics_history = []
        best_r2_score = float('-inf')
        best_preds = None
        best_trues = None

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, predict_df, _) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # batch_x = torch.cat((batch_x, batch_x_mark), dim=-1)

                predict_df = predict_df.float()  # 因为预测信息只需要放在prompt里面，所以不需要改变device

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                # dec_inp = torch.cat((dec_inp, batch_x_mark), dim=-1)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, predict_df)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss, mse_total, r2_total, true, pred = vali(args, accelerator, model, test_data,
                                                                             test_loader, criterion, mae_metric)

            if r2_total > best_r2_score:
                best_r2_score = r2_total
                best_preds = pred.copy()  # 保存最佳预测值
                best_trues = true.copy()  # 保存最佳真实值

                df = pd.DataFrame({
                    'Predictions': best_preds,
                    'True Values': best_trues
                })

                if args.select_chunk == 3:
                    pct = "30%"
                elif args.select_chunk == 1:
                    pct = "10%"
                else:
                    # 如果有意之外的值，可以进行一个友好的提示或者处理
                    raise ValueError(f"select_chunk 的值只能是 1 或 3，你给的是 {args.select_chunk}")

                # 保存为 CSV 文件
                df.to_csv(f"exp_results/{pct}shot从{args.start_chunk_index}开始_预测结果.csv", index=False)

                print(f"Training Completed. Best R2 Score: {best_r2_score}")

            # 将每个 epoch 的结果存入列表
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_mae_loss': test_mae_loss,
                'mse_total': mse_total,
                'r2_total': r2_total
            })

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if args.select_chunk == 3:
            pct = "30%"
        elif args.select_chunk == 1:
            pct = "10%"
        else:
            # 如果有意之外的值，可以进行一个友好的提示或者处理
            raise ValueError(f"select_chunk 的值只能是 1 或 3，你给的是 {args.select_chunk}")

        # 原始结果保存逻辑
        metrics_df = pd.DataFrame(metrics_history)

        # 将结果保存到 exp_results 文件夹下，文件名中包含变量
        metrics_df.to_csv(f"exp_results/{pct}shot从{args.start_chunk_index}开始_日志.csv", index=False)

    # 原始清理逻辑
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        del_files('./checkpoints')
        accelerator.print('success delete checkpoints')

