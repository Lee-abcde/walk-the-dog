import os
import os.path as osp
import random

from tqdm import tqdm

import Library.Utility as utility
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
from models import VQ as model
from models import phase_decoder as phase_decoder_model
import Plotting as plot
from modules import save_manifold
from Library.Utility import Item, ItemNumpy

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from option import TrainVQOptionParser
from utils.loss_recorder import LossRecorder

from dataset import create_dataset_from_args
import matplotlib.pyplot as plt

from utils.vq_plotting import plot_embedding, plot_usage_freq
from trainers.phase_decoder import PhaseDecoderTrainer


def prepare_phase_decoder(args, args_phase_decoder, motion_data):
    # 该函数的作用是：
    # 创建相位解码器模型，并为其指定数据和输出特征。
    # 处理模型的名称和输出维度信息，用于在训练过程中记录和配置模型。
    # 创建并返回一个相位解码器训练器，它将管理解码器的训练流程。
    network = phase_decoder_model.create_model_from_args2(args, motion_data)
    phase_model_name = args.save[-7:].replace('/', '-')
    output_channel_names = args.needed_channel_names_phase_decoder
    output_feature_dims = motion_data.get_feature_dim_by_names(output_channel_names)
    trainer = PhaseDecoderTrainer(args.n_latent_channel, network, args_phase_decoder,  phase_model_name,
                                  output_channel_names, motion_data.name, output_feature_dims,
                                  args.lr_phase_decoder)
    return trainer


def nan_in_grad(model: torch.optim.Optimizer):
    for group in model.param_groups:
        for p in group['params']:
            if p.requires_grad and torch.isnan(p.grad).any():
                return True
    return False


def main():
    option_parser = TrainVQOptionParser()
    args = option_parser.parse_args()

    # the input storage path
    Save = args.save
    utility.MakeDirectory(Save)
    # write args.txt
    with open(osp.join(Save, "args.txt"), "w") as file:
        file.write(option_parser.text_serialize(args))
    args = option_parser.post_process(args)

    # 创建log folder
    log_dir = osp.join(Save, 'log')
    if os.path.exists(log_dir) and 'test' not in log_dir:
        print('log dir exists, remove it [y/n]?')
        if input() != 'y':
            print('exit')
            return
    if osp.exists(log_dir):
        os.system(f'rm -rf {log_dir}')
    # PyTorch 中 TensorBoard 的一个组件，主要用于记录和可视化训练过程中的数据
    summary_writer = SummaryWriter(log_dir)
    # 自定义了一个类LossRecorder， SummaryWriter是其中的一个变量
    loss_recorder = LossRecorder(summary_writer)

    plot_cnt = 0

    # load motion data
    motion_datas = create_dataset_from_args(args)
    lengths = [len(data) for data in motion_datas] #列表推导式来计算每个数据集的长度
    idx = np.argsort(lengths) #将返回一个包含索引的数组，按对应的长度升序排列。
    smallest_idx = idx[0] #获取第一个索引，即长度最小的数据集的索引。

    plt.ioff() # 关闭 Matplotlib 的交互模式
    fig1, ax1 = plt.subplots(6,1) #创建一个包含 6 行 1 列的子图
    fig2, ax2 = plt.subplots(args.phase_channels,5)
    if args.phase_channels == 1:
        ax2 = ax2[None]
    fig3, ax3 = plt.subplots(1,2)
    fig4, ax4 = plt.subplots(2,1)

    figs = [fig1, fig2, fig3, fig4]
    axs = [ax1, ax2, ax3, ax4]

    for i in range(args.use_vq):
        n_fig, n_ax = plt.subplots(2, 2)
        n_ax[1, 1].axes.xaxis.set_visible(False)
        n_ax[1, 1].axes.yaxis.set_visible(False)

        figs.append(n_fig)
        axs.append(n_ax)

    dist_amps = []
    dist_freqs = []
    loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=args.plotting_interval)

    # Build network model
    networks, VQs = model.create_model_from_args(args, motion_datas) # VQ是ResidualVectorQuantizer
    networks = utility.ToDevice(networks)
    VQs = utility.ToDevice(VQs)

    params = sum([list(n.parameters()) for n in networks], []) #networks 可能是一个由多个模型组成的列表。n.parameters() 返回模型 n 的所有参数（权重和偏差等）。
    params += list(VQs.parameters()) #这一行将量化器 VQs 的参数也添加到 params 列表中，这样所有模型和量化器的参数都被收集到了 params 列表中。

    # Setup optimizer and loss function
    optimizer = adamw.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=args.batch_size, epoch_size=len(motion_datas[smallest_idx]), restart_period=args.restart_period, t_mult=args.restart_mult, policy="cosine", verbose=True)
    data_loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                               pin_memory=True) for data in motion_datas]

    criteria = {'rec': torch.nn.MSELoss()}

    # train decoder
    if args.train_phase_decoder:
        phase_decoder_trainers = []
        for data in motion_datas:
            phase_decoder_trainers.append(prepare_phase_decoder(args, args, data))
    else:
        phase_decoder_trainers = []

    nan_count = 0

    #start to train
    for epoch in range(args.epochs):
        scheduler.step() #每个epoch调用一次学习率调度器
        loop = tqdm(range(len(data_loaders[smallest_idx]))) #用tqdm创建一个循环进度条，循环次数等于最小数据集的批次长度。
        its = [iter(loader) for loader in data_loaders]
        VQs.clear_buffer()
        for n_iters in loop:
            loss_totals = [] #初始化一个空列表，用于存储每个类别的损失
            for class_idx in range(len(data_loaders)):
                #Run model prediction
                motion_data = motion_datas[class_idx] #获取当前类别的数据对象
                VQs.set_caller(class_idx)
                network = networks[class_idx]
                train_batch = next(its[class_idx]) #从当前类别的数据迭代器中获取下一批次的数据
                network.train()
                VQs.train()
                losses = {}
                train_batch = utility.ToDevice(train_batch)
                pae_input = motion_data.get_feature_by_names(train_batch, args.needed_channel_names) # pae = phase autoEncoder
                yPred, latent, signal, params, vq_info = network(pae_input)

                # train phase decoder
                if args.train_phase_decoder:
                    trainer = phase_decoder_trainers[class_idx]
                    gt = motion_data.get_feature_by_names(train_batch, args.needed_channel_names_phase_decoder)
                    if args.decoder_before_quantization:
                        input = params[5]  # manifold_ori
                    else:
                        input = params[4]  # manifold
                    input = input.permute(0, 2, 1)
                    input = input.reshape(-1, input.shape[-1])
                    gt = gt.permute(0, 2, 1)
                    gt = gt.reshape(-1, gt.shape[-1])
                    trainer.zero_grad()
                    trainer.forward(input, gt)
                    loss_phase_decoder = trainer.loss_total
                    trainer.record_loss(loss_recorder, f'{class_idx}/phase_decoder_')
                else:
                    loss_phase_decoder = 0.

                # Compute loss and train
                losses['rec'] = criteria['rec'](yPred, pae_input)

                if args.use_vq:
                    losses['vq'] = vq_info[0]
                    perplexities = vq_info[1]
                    for i, p in enumerate(perplexities):
                        loss_recorder.add_scalar(f'{class_idx}/perplexity_{i}', p)

                _a_ = Item(params[2]).reshape(-1, args.phase_channels).numpy()
                for i in range(_a_.shape[0]):
                    dist_amps.append(_a_[i, :]) #将每行振幅数据追加到dist_amps列表中，dist_amps可能是一个存储振幅分布的列表。
                while len(dist_amps) > 10000:
                    dist_amps.pop(0)

                _f_ = Item(params[1]).reshape(-1, args.phase_channels).numpy() #将params[1]（可能是频率信息）转换为特定形状的NumPy数组。
                for i in range(_f_.shape[0]):
                    dist_freqs.append(_f_[i, :])
                while len(dist_freqs) > 10000:
                    dist_freqs.pop(0)

                loss_total = sum([losses[k] * getattr(args, f'lambda_{k}') for k in losses])
                loss_total = loss_total + loss_phase_decoder

                for k in losses:
                    loss_recorder.add_scalar(f'{class_idx}/loss_{k}', losses[k].item())
                loss_recorder.add_scalar(f'{class_idx}/loss_total', loss_total.item())

                loss_descript = ' '.join([f'{k}: {v.item():.4f}' for k, v in losses.items()])
                loss_descript = f'total: {loss_total.item():.4f} ' + loss_descript
                loop.set_description(loss_descript)

                loss_totals.append(loss_total)

                loss_history.Add(
                    (Item(losses['rec']).item(), "Reconstruction Loss")
                )

            loss_total = sum(loss_totals) / len(loss_totals)
            optimizer.zero_grad()
            loss_total.backward()

            if not (torch.isnan(loss_total) or nan_in_grad(optimizer)):
                VQs.reinitialize()
                optimizer.step()
                for trainer in phase_decoder_trainers:
                    trainer.step()
            else:
                nan_count += 1
            loss_recorder.add_scalar(f'nan_count', nan_count)
            scheduler.batch_step()

            # Start Visualization Section
            if loss_history.Counter == 0 or args.debug:
                class_idx = random.randint(0, len(data_loaders)-1)
                motion_data = motion_datas[class_idx]
                network = networks[class_idx]

                VQs.set_caller(-1)

                network.eval()
                VQs.eval()

                test_sample = motion_data.sample_continuous_test_window()
                test_sample = motion_data.get_feature_by_names(test_sample, args.needed_channel_names)
                test_sample = utility.ToDevice(test_sample)
                yPred, latent, signal, params, _ = network(test_sample)

                frames = motion_data.frames_per_window

                plot.Functions(ax1[0], Item(test_sample[0]), -1.0, 1.0, -5.0, 5.0, title=f"Motion Curves {network.n_input_channels}x{frames}", showAxes=False)
                plot.Functions(ax1[1], Item(latent[0].squeeze()), -1.0, 1.0, -2.0, 2.0, title=f"Latent Convolutional Embedding {args.phase_channels}x{frames}", showAxes=False)
                plot.Circles(ax1[2], Item(params[0][0]), Item(params[2][0]), title=f"Learned Phase Timing {args.phase_channels}x2", showAxes=False)
                plot.Functions(ax1[3], Item(signal[0, 0]), -1.0, 1.0, -2.0, 2.0, title=f"Latent Parametrized Signal {args.phase_channels}x{frames}", showAxes=False)
                plot.Functions(ax1[4], Item(yPred[0].squeeze()), -1.0, 1.0, -5.0, 5.0, title=f"Curve Reconstruction {network.n_input_channels}x{frames}", showAxes=False)
                plot.Function(ax1[5], [Item(test_sample[0]).reshape(-1), Item(yPred[0]).reshape(-1)], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title=f"Curve Reconstruction (Flattened) 1x{network.n_input_channels * frames}", showAxes=False)
                plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

                for i in range(args.phase_channels):
                    phase = params[0][:,i].unsqueeze(1)
                    freq = params[1][:,i].unsqueeze(1)
                    amps = params[2][:,i].unsqueeze(1)
                    offs = params[3][:,i].unsqueeze(1)
                    plot.Phase1D(ax2[i,0], Item(phase), Item(amps), color=(0, 0, 0), title=("1D Phase Values" if i==0 else None), showAxes=False)
                    plot.Phase2D(ax2[i,1], Item(phase), Item(amps), title=("2D Phase Vectors" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,2], Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,3], Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if i==0 else None), showAxes=False)
                    plot.Functions(ax2[i,4], Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if i==0 else None), showAxes=False)
                
                # Visualization
                pca_indices = []
                pca_batches = []
                pivot = 0
                for i in range(args.pca_sequence_count):
                    test_sample = motion_data.sample_continuous_test_window()
                    test_sample = motion_data.get_feature_by_names(test_sample, args.needed_channel_names)
                    test_sample = utility.ToDevice(test_sample)
                    _, _, _, params, info = network(test_sample)
                    p = Item(params[0]).squeeze()
                    state = Item(info[2])
                    manifold = model.get_phase_manifold(state, 2 * np.pi * p[:, None, None])[0].squeeze(-1)
                    pca_indices.append(pivot + np.arange(motion_data.window_size_test))
                    pca_batches.append(manifold)
                    pivot += motion_data.window_size_test
                plot.PCA2D(ax4[0], pca_indices, pca_batches, f"Phase Manifold ({args.pca_sequence_count} Random Sequences)")

                # Plot VQ
                for i in range(VQs.num_steps):
                    ax5 = axs[4 + i]
                    VQ = VQs.vqs[i]
                    embedding = ItemNumpy(VQ.embedding.weight)

                    usage = VQ.usage
                    log_usage = np.log(usage.sum(axis=0) + 1e-5)
                    usage = (usage > 0).astype(np.int32)
                    usage = sum([(2 ** i) * usage[i] for i in range(usage.shape[0])])
                    plot_embedding(ax5[0, 0], embedding, usage)
                    plot_embedding(ax5[1, 0], embedding, log_usage / log_usage.max())
                    plot_usage_freq(ax5[0, 1], usage)

                dpis = [100] * 4 + [300] * args.use_vq
                for i, fig in enumerate(figs):
                    fig.suptitle(f"Data class: {class_idx}")
                    img = utility.fig2tensor(fig, dpi=dpis[i])
                    summary_writer.add_image(f'Figure {i}', img, plot_cnt)
                plot_cnt += 1

            if args.debug:
                break

        for i, network in enumerate(networks):
            torch.save(network.state_dict(), f'{Save}/{epoch+1}_{i}_{args.phase_channels}Channels.pt')
            if args.train_phase_decoder:
                phase_decoder_trainers[i].export_model(Save, epoch, prefix=f'{motion_datas[i].name}_')
        torch.save(VQs.state_dict(), f'{Save}/{epoch+1}_{args.phase_channels}Channels_VQ.pt')

        VQs.clear_usage()

        print('Epoch', epoch+1, loss_history.CumulativeValue())
        loss_recorder.epoch()

        # Save Phase Parameters
        print("Saving Parameters")
        for network in networks:
            network.eval()
        VQs.eval()

        for i, network in enumerate(networks):
            filename_npy = osp.join(Save, f'Manifolds_{i}_{epoch+1}.npz')
            save_manifold(network, motion_datas[i], filename_npy, args)


if __name__ == '__main__':
    main()
