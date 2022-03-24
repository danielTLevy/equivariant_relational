# -*- coding: utf-8 -*-

from data.synthetic_heterogeneous import SyntheticHG

#%%
import sys
sys.path.append('../../')
sys.path.append('../')

import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hgb.NC.EquivHGNet import EquivHGNet
from utils import get_hyperparams, set_seed, select_features, f1_scores
from src.utils import count_parameters
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SyntheticHG()
    dl = SyntheticHG(args.n_ents, args.n_rels, args.data_embed,
                     args.n_instances, args.sparsity, args.p_het,
                     gen_links=args.gen_links)
    dl.make_node_classification_task(args.n_classes, args.pct_test,
                                     args.pct_val, args.node_label)
    data = dl.data
    target_entity_id = 0
    target_entity = dl.schema.entities[target_entity_id]

    data, in_dims = select_features(dl.data, dl.schema, args.feats_type, target_entity_id)
    labels = torch.LongTensor(dl.labels).to(device)
    train_idx = np.sort(dl.train_idx)
    val_idx = np.sort(dl.val_idx)
    test_idx = np.sort(dl.test_idx)
    
    data = data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    data_target = dl.data_target.to(device)

    net = EquivHGNet(dl.schema, in_dims,
                        layers = args.layers,
                        in_fc_layer=args.in_fc_layer,
                        fc_layers=args.fc_layers,
                        activation=eval('nn.%s()' % args.act_fn),
                        final_activation = nn.Identity(),
                        target_entities=[target_entity],
                        dropout=args.dropout,
                        output_dim=dl.n_classes,
                        norm=args.norm,
                        pool_op=args.pool_op,
                        norm_affine=args.norm_affine,
                        norm_out=args.norm_out,
                        residual=args.residual,
                        mid_fc_layer=args.mid_fc_layer)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="EquivariantHGN_Synth_NC",
            entity='danieltlevy')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    print("Number of parameters: {}".format(count_parameters(net)))

    run_name = str(args.run)
    if args.wandb_log_run and wandb.run.name is not None:
        run_name = run_name + '_' + str(wandb.run.name)

    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f"checkpoint/checkpoint_{run_name}.pt"

    print("Checkpoint Path: " + checkpoint_path)
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    # training loop
    net.train()
    val_micro_best = 0
    for epoch in progress:
        # training
        net.train()
        optimizer.zero_grad()
        logits = net(data, indices_identity, indices_transpose,
                     data_target).squeeze()
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        train_micro, train_macro = f1_scores(logits[train_idx],
                                                 labels[train_idx])
        with torch.no_grad():
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), micr=train_micro)
            wandb_log = {'Train Loss': train_loss.item(),
                         'Train Micro': train_micro,
                         'Train Macro': train_macro}
            if epoch % args.val_every == 0:
                # validation
                net.eval()
                logits = net(data, indices_identity, indices_transpose, data_target).squeeze()
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
                val_micro, val_macro = f1_scores(logits[val_idx],
                                                         labels[val_idx])
                print("\nVal Loss: {:.3f} Val Micro-F1: {:.3f} \
Val Macro-F1: {:.3f}".format(val_loss, val_micro, val_macro))
                wandb_log.update({'Val Loss': val_loss.item(),
                                  'Val Micro-F1': val_micro, 'Val Macro-F1': val_macro})
                if val_micro > val_micro_best:

                    val_micro_best = val_micro
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_micro': train_micro,
                        'train_macro': train_macro,
                        'val_loss': val_loss.item(),
                        'val_micro': val_micro,
                        'val_macro': val_macro
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_micro_best"] = val_micro
                        wandb.summary["val_macro_best"] = val_macro
                        wandb.summary["val_loss_best"] = val_loss.item()
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.summary['train_micro_best'] = train_micro
                        wandb.summary['train_macro_best'] = train_macro
                        wandb.save(checkpoint_path)

            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log, step=epoch)


    # testing on test set
    if args.evaluate:
        pass
    wandb.finish()

#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    #%%
    run_model(args)
