# %%
import torch
import esm

# %%  Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# %% Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


# %% 
model

# %% Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# %% Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# %% Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq)
    plt.show()

# %% 

# %% Looking at the token representation maps 
for (_, seq), tokens_len, representations in zip(data, batch_lens, results["representations"]):
    print(representations)
# %%
for i in range(4): 
    plt.imshow(token_representations[i,:,:])  
    plt.show()
# %%
for i in range(4):
    print(len(data[i][1]))

# %%
token_representations.shape

# %% Performing pca on the token_representations 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(token_representations[0,:,:])

# %% Visualzing the pca results 
import matplotlib.pyplot as plt
plt.scatter(pca.components_[0,:],pca.components_[1,:])
plt.show()

# %% Let's see which residues are "special" according to the PCA 
import numpy as np
residue_pca = pca.components_
residue_pca.shape

# %% Let's now fine-tune the ESM model on the protein2 sequence

num_iters = 3
learning_rate = 0.001

# we are using protein sequence 2 as our training data. 
# we will start by overfitting on this 1 example as a test run. 

for i in range(num_iters):
    # forward pass 
    # compute loss
    batch_labels, batch_strs, batch_tokens = batch_converter([data[1]])
    output = model(batch_tokens, repr_layers=[33], return_contacts=True)
    logits = output["logits"]

    # compute loss
    loss = torch.nn.functional.cross_entropy(logits, batch_labels)

    print("Iteration: ", i, " Loss: ", loss.item())
    # backward pass 
    # update parameters 
    pass

# %%
