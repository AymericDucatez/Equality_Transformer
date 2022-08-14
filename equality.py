import torch
import math
import random
import encoder
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


ap = argparse.ArgumentParser()
ap.add_argument('--train_length', dest='train_length', type=int, default=100)
ap.add_argument('--test_length', dest='test_length', type=int, default=100)
ap.add_argument('--epochs', dest='epochs', type=int, default=500)
ap.add_argument('--steps', dest='steps', type=int, default=100)
ap.add_argument('--layers', dest='layers', type=int, default=2)
ap.add_argument('--heads', dest='heads', type=int, default=2)
ap.add_argument('--d_model', type=int, default=16) # i.e. the word encoding dimension
ap.add_argument('--d_ffnn', type=int, default=64) # hidden units in FFNN
ap.add_argument('--scaled', type=bool, default=False, help='log-length scaled attention')
ap.add_argument('--eps', type=float, default=1e-5, help='Value added to denominator in layer normalization')
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

# Initialize the random number generators for reproducibility
random.seed(123)
torch.manual_seed(123)
np.random.seed(123) # is this one needed?

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        super().__init__()
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)

        if scaled:
            # The difference between ScaledTransformerEncoderLayer and 
            # TransformerEncoderLayer is the scaling of the input_sequence, i.e.
            # (character + positional embedding) sequence, used for the query 
            # calculation in the attention block, by a factor log(len(input_sequence))
            encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = encoder.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        # layer-normalization after residual connections is the default,
        # with the value of norm1.eps and norm2.eps being consistent also as
        # default, and its value being 1e-5
        # norm1.eps is the value for layer-norm after the attention residual 
        # connection and norm2.eps is the value after the FFNN residual connect.
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps 
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_layer = torch.nn.Linear(d_model, 1)

    def forward(self, w):
        # w is of dimension n x size and pos_encoding of n x size, where size = d_model
        # here word_embedding "embeds" a sequence of characters into a sequence
        # of vectors
        x = self.word_embedding(w)
        # unsqueeze to match the input dims [seq_len, batch_size, embedding_dim]
        # so as to have [[[...]], ...,[[...]]]  since batch_size = 1
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        # squeeze again to remove batch_size parenthesis and select the last
        # character of the sequence "2" (i.e. CLS)
        z = self.output_layer(y[-1])
        return z

model = Model(3, args.layers, args.heads, args.d_model, args.d_ffnn, args.scaled, args.eps)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)

# Investigating trainable model parameters
#params_dict = model.state_dict()
#print(params_dict.items())
# or
#for name, param in model.named_parameters():
#   if param.requires_grad:
#        print(name, param.data)

def check_equality_label(w):
    n = 0
    for i in range(0,len(w)-1):
        n += w[i]*2-1
    return n == 0

'''
TODO
- fix the step thing
- print the predictions and not if it's correct or not
- switch between .train() and .eval()
- fix training/test loops according to tutorial

'''

n = args.train_length
m = args.test_length
num_datapoints = args.steps
TEST_SPLIT = 0.2 # this is the validation split and not a held-out set
BATCH_SIZE = 1


print('Training string length',n)
print('Test string length', m)

# Generating the balanced dataset
training_data = []
training_labels = []
true_counter = 0
false_counter = 0

print('Balanced train set generation STARTED.')
while (true_counter + false_counter) < num_datapoints:
    w = torch.tensor([random.randrange(2) for i in range(n)]+[2])
    label = check_equality_label(w)
    if label == True and true_counter < num_datapoints/2:
        training_data.append(w)
        training_labels.append(label)
        true_counter += 1
    elif label == False and false_counter < num_datapoints/2:
        training_data.append(w)
        training_labels.append(label)
        false_counter += 1
print('Balanced train set generation COMPLETED.')

test_data = []
test_labels = []
true_counter = 0
false_counter = 0

print('Balanced test set generation STARTED.')
while (true_counter + false_counter) < num_datapoints:
    w = torch.tensor([random.randrange(2) for i in range(m)]+[2])
    label = check_equality_label(w)
    if label == True and true_counter < num_datapoints/2:
        test_data.append(w)
        test_labels.append(label)
        true_counter += 1
    elif label == False and false_counter < num_datapoints/2:
        test_data.append(w)
        test_labels.append(label)
        false_counter += 1
print('Balanced test set generation COMPLETED.')

#X_train, X_test, y_train, y_test = train_test_split(
#    training_data, training_labels, test_size=TEST_SPLIT, shuffle=True, 
#    random_state=123)



class CustomDataset(Dataset):
    def __init__(self, X_list, y_list):
            self.data = X_list
            self.labels = y_list 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


training_dataset = CustomDataset(training_data, training_labels)
test_dataset = CustomDataset(test_data, test_labels)

train_dataloader = DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True)


for epoch in (range(args.epochs)):
    print(epoch)
    train_loss = 0
    train_steps = 0
    train_correct = 0
    train_in_lang = 0
    train_not_lang = 0
    true_label_ls = []
    correct_prediction_ls = []
    
    
    for batch, (X, y) in enumerate(train_dataloader):
        w = X
        if BATCH_SIZE==1:
            w = torch.flatten(w)
        label = y
        if label:
            train_in_lang += 1
        elif not label:
            train_not_lang += 1
        output = model(w)
        if not label: output = -output
        if output > 0: train_correct += 1
        loss = -log_sigmoid(output)
        train_loss += loss.item()
        train_steps += 1
        optim.zero_grad()
        loss.backward()
        optim.step()

    train_frac_in = train_in_lang/(train_in_lang + train_not_lang)

    test_loss = 0
    test_steps = 0
    test_correct = 0
    test_in_lang = 0
    test_not_lang = 0

    for batch, (X, y) in enumerate(test_dataloader):
        w = X
        if BATCH_SIZE==1:
            w = torch.flatten(w)
        label = y
        true_label_ls.append(int(label.item()))
        if label:
            test_in_lang += 1
        elif not label:
            test_not_lang += 1
            
        output = model(w)
        if not label: output = -output
        if output > 0:
            test_correct += 1
            correct_prediction_ls.append(1)
        else:
            correct_prediction_ls.append(0)
        
        loss = -log_sigmoid(output)
        test_loss += loss.item()
        test_steps += 1
    
    test_frac_in = test_in_lang/(test_in_lang + test_not_lang)
    print(f'Fractions of strings in formal language - training: {train_frac_in}, test: {test_frac_in}')

    print(f'train_length={args.train_length} train_ce={train_loss/train_steps/math.log(2)} train_acc={train_correct/train_steps} test_ce={test_loss/test_steps/math.log(2)} test_acc={test_correct/test_steps}', flush=True)


