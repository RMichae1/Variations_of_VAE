
import torch
import math
import pickle
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from helper_functions import *
from models import *
from data_handler import *
from training import *
from Bio import SeqIO
from torch import optim
import datetime


# Device
device = 'cuda'

# determine VAE
epochs = 15
offset = 4
latent_dim = 30
name = 'ubqt'
extra = 'test'
split_by_DI = False
pos_per_fold = pos_per_fold_assigner(name.lower())
if name == 'blat':
    name = name.upper()
   
df = pickle.load( open(name.lower()+'_data_df.pkl', "rb" ) )
query_seqs = df['seqs'][0]
assay_df = df.dropna(subset=['assay']).reset_index(drop=True)
y = assay_df['assay']

random_weighted_sampling = True
use_sparse_interactions = True
use_bayesian = True
use_param_loss = True
batch_size = 100
use_cluster='regular'
assay_label = ['assay']
labels = [None]

assay_index = df.index
train_idx = [i for i in df.index if i not in assay_index.values]
all_data, train_data, val_data = get_datasets(data=df,
                                              train_ratio=1,
                                              device = device,
                                              SSVAE=0,
                                              SSCVAE=0,
                                              CVAE=0,
                                              regCVAE=0,
                                              assays_to_include=['assay'],
                                              train_index=train_idx,
                                              train_with_assay_seq=0,
                                              only_assay_seqs=0)

# prep downstream data
def onehot_(arr):
    return F.one_hot(torch.stack([torch.tensor(seq, device='cpu').long() for seq in arr]), num_classes=max(IUPAC_SEQ2IDX.values())+1).float().flatten(1)

X_all_torch = torch.from_numpy(np.vstack(df['seqs'].values))
X_labelled_torch = torch.from_numpy(np.vstack(assay_df['seqs'].values))

# Construct dataloaders for batches
train_loader = get_protein_dataloader(train_data, batch_size = batch_size, shuffle = False, random_weighted_sampling = random_weighted_sampling)
val_loader = get_protein_dataloader(val_data, batch_size = 100)
print("Data loaded!")

# log_interval
log_interval = list(range(1, epochs, 3))

# define input and output shape
data_size = all_data[0][0].size(-1) * alphabet_size
label_pred_layer_sizes = [0]
z2yalpha = 0.01 * len(all_data)
seq2yalpha = 0.01 * len(all_data)
model = VAE_bayes(
    [data_size] + [1500, 1500, latent_dim, 100, 2000] + [data_size],
    alphabet_size,
    z_samples = 1,
    dropout = 0,
    use_bayesian = use_bayesian,
    use_param_loss = use_param_loss,
    use_sparse_interactions = use_sparse_interactions,
    rws=random_weighted_sampling,
    conditional_data_dim=0,
    SSVAE = 0,
    SSCVAE = 0,
    CVAE=0,
    VAE=1,
    regCVAE=0,
    multilabel=0,
    seq2yalpha=seq2yalpha,
    z2yalpha=z2yalpha,
    label_pred_layer_sizes = label_pred_layer_sizes,
    pred_from_latent=0,
    pred_from_seq=0,
    warm_up = 0,
    batchnorm=0,
    device = device
)

optimizer = optim.Adam(model.parameters())

print(model.summary())
print(model)

date = 'D'+str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)
time = 'T'+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
date_time = date+time

# train_indices, val_indices, test_indices = positional_splitter(assay_df, query_seqs, val=True, offset = 4, pos_per_fold = pos_per_fold, 
#                                             split_by_DI = False)

results_dict = defaultdict(list)
best_downstream_loss = np.inf
overall_start_time = datetime.datetime.now()

VALIDATION_EPSILON = 10e-3 # convergence rate of validation
validation_errors = [100000000] # initial error very high, mitigates early indexing issues
overfitting_patience = 10 # validation intervals

for epoch in range(1, epochs + 1):
    if torch.cuda.is_available():
        model = model.cuda().float()
    start_time = datetime.datetime.now()
    train_loss, train_metrics, px_z, aux_loss = train_epoch(epoch = epoch, model = model, optimizer = optimizer, scheduler=None, train_loader = train_loader)

    loss_str = "Training"
    loss_value_str = f"{train_loss:.5f}"
    val_str = ""

    results_dict['epochs'].append(epoch)
    results_dict['nll_loss_train'].append(train_metrics["nll_loss"])
    results_dict['kld_loss_train'].append(train_metrics["kld_loss"])
    results_dict['param_kld_train'].append(train_metrics["param_kld"])
    results_dict['total_train_loss'].append(train_loss)
    results_dict['aux_loss'].append(aux_loss)

    # print status
    print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Recon loss: {train_metrics['nll_loss']:.5f} KLdiv loss: {train_metrics['kld_loss']:.5f} Auxiliary loss: {aux_loss} Param loss: {train_metrics['param_kld']:.5f} {val_str}Time: {datetime.datetime.now() - start_time}", end="\n\n")
    if epoch in log_interval:
        model.eval()
        val_loss, _, _, val_aux_loss = train_epoch(epoch = epoch, model = model, 
                                    optimizer = optimizer, scheduler=None, train_loader = val_loader)
        val_diff = np.abs(validation_errors[-1]-val_loss)
        print(f"Validation step: {epoch} - loss: {val_loss}, abs.diff={val_diff}, val_aux_loss: {val_aux_loss}")
        # TODO: setup torch saving of model weights during validation
        if val_diff <= VALIDATION_EPSILON or overfitting_patience == 0:
          print(f"ENDING TRAINING!")
          if val_diff <= VALIDATION_EPSILON:
            print("Validation delta reached...")
          if overfitting_patience == 0:
            print("overfitted after 5 validation steps")
          break
        if val_loss > validation_errors[-1]:
          print("Overfitted step... ")
          overfitting_patience -= 1
        validation_errors.append(val_loss)
        model.train()

print('total epoch time', datetime.datetime.now()-start_time)
        
with torch.no_grad():
    print('Saving...')
    pickle.dump( results_dict, open('final/VAE_'+extra+date_time+'_'+name+'_'+str(latent_dim)+'dim_final_results_dict.pkl', "wb" ) )
print('Total time: ', datetime.datetime.now() - overall_start_time)
