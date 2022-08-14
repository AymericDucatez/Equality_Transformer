import torch
import encoder
import math
import random
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=100)
ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--big', dest='big', type=float, default=1.)
args = ap.parse_args()
alphabet = ["0", "1", "$"]
alphabet_index = {a:i for i,a in enumerate(alphabet)}
log_sigmoid = torch.nn.LogSigmoid()

def check_equality_label(w):
    n = 0
    for i in range(1,len(w)):
        n += w[i]*2-1
    return n == 0

class FirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(8, 1, 2, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # Only one head, that counts the 0 and 1s
            # W^Q
            [[0]*8]*8 +
            # W^K
            [[0]*8]*8 +
            # W^V
            [[1,0,0,0,0,0,0,0],   # count 0s
             [0,1,0,0,0,0,0,0]]+   # count 1s
            [[0]*8]*6,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(24))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*8]*3 +
            [[1,0,0,0,0,0,0,0],   # put new values into dims 4,5
             [0,1,0,0,0,0,0,0]] +
            [[0]*8]*3,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(8))

        self.linear1.weight = torch.nn.Parameter(torch.tensor(
            [[0]*8]*5+
			[[0,0,0,1,-1,0,0,0],
			[0,0,0,-1,1,0,0,0]]+
			[[0]*8], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(8))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0]*8]*5+[[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],
            [0,0,0,0,0,1,1,0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(8))
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, verbose = False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src1 = self.dropout(self.activation(self.linear1(src)))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))+src
        return src2
    
class MyTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstLayer()
        ])
        self.num_layers = len(self.layers)
        self.norm = None

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.word_embedding = torch.eye(3, 8)
        self.transformer_encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(8, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,-1]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        x = self.word_embedding[w]
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z


length_ls = []
accuracy_ls = []
ce_ls = []
for n in tqdm(range(1,args.length+1)):
    model = Model()

    loss = 0
    total = 0
    correct = 0
    for step in range(args.steps):
        w = torch.tensor([alphabet_index['$']] + [alphabet_index[str(random.randrange(2))] for i in range(n)])
        label = check_equality_label(w)
        output = model(w)

        if (output != 0 and not(label)) or (output == 0 and label):
            correct += 1
        #else : 
            #print(w)
            #print(output)
            #print(label)

        total += 1
        loss -= log_sigmoid(output).item()
    #print(f'length={n} ce={loss/total/math.log(2)} acc={correct/total}')
    length_ls.append(n)
    ce_ls.append(loss/total/math.log(2))
    accuracy_ls.append(correct/total)

# Figure    
fig, (ax_1, ax_2) = plt.subplots(2,1)

fig.subplots_adjust(hspace=0.5)

# Cross-entropy plot
ax_1.plot(length_ls, ce_ls)
ax_1.set_xlabel('string legnth n')
ax_1.set_ylabel('cross-entropy (bits)')
ax_1.set(ylim = (0, 1.5), yticks=[0, 0.5, 1, 1.5])

# Accuracy plot
ax_2.plot(length_ls, accuracy_ls, 'r')
ax_2.set_xlabel('string legnth n')
ax_2.set_ylabel('accuracy')
ax_2.set(ylim = (0, 1.2), yticks=[0, 0.5, 1])

fig.suptitle('EQUALITY')

plt.savefig('equality_exact_ce_accuracy.png')

