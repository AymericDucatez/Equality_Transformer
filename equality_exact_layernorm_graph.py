import torch
import encoder
import math
import random
import sys
import argparse
import copy
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
sigmoid = torch.nn.Sigmoid()

def check_equality_label(w):
    n = 0
    for i in range(1,len(w)):
        n += w[i]*2-1
    return n == 0

class FirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(16, 1, 2, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # Only one head, that counts the 0 and 1s
            # W^Q
            [[0]*16]*16 +
            # W^K
            [[0]*16]*16 +
            # W^V
            [[1,0,0,0,0,0,0,0]+[0]*8,   # count 0s
             [0,1,0,0,0,0,0,0]+[0]*8]+   # count 1s
            [[0]*16]*14,
            dtype=torch.float))


        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(48))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*16]*3 +
            [[1,0,0,0,0,0,0,0]+[0]*8,   # put new values into dims 4,5
             [0,1,0,0,0,0,0,0]+[0]*8] +
            [[0]*16]*6 +
            [[-1,0,0,0,0,0,0,0]+[0]*8, 
            [0,-1,0,0,0,0,0,0]+[0]*8]+
            [[0]*16]*3,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(16))

        self.linear1.weight = torch.nn.Parameter(torch.tensor(
            [[0]*16]*5+
			[[0,0,0,1,-1,0,0,0]+[0]*8,
			[0,0,0,-1,1,0,0,0]+[0]*8]+
			[[0]*16]*9, dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(16))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0]*16]*5 + [[0,0,0,0,0,1,0,0]+[0]*8,[0,0,0,0,0,0,1,0]+[0]*8]+
            [[0,0,0,0,0,1,1,0]+[0]*8]+ 
            [[0]*16]*5 + [[0,0,0,0,0,-1,0,0]+[0]*8,[0,0,0,0,0,0,-1,0]+[0]*8]+
            [[0,0,0,0,0,-1,-1,0]+[0]*8],
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(16))

    def layernorm(self,x):
        y = x
        for i in range(len(y)) :
            mean = 0
            aux = 0
            for sc in y[i][0] :
                mean += sc
                aux += sc**2
            mean /= len(y[i][0])
            aux /= len(y[i][0])
            var = aux-(mean**2)
            for j in range(len(y[i][0])):
                y[i][0][j]=(y[i][0][j]-mean)/math.sqrt(var)  
        return y

    def forward(self, src, src_mask=None, src_key_padding_mask=None, verbose = False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.layernorm(src)
        src1 = self.dropout(self.activation(self.linear1(src)))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.layernorm(src2+src)
        return src2
    

class SecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(16, 1, 2, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # Only one head, that doesn't do anything
            # W^Q
            [[0]*16]*16 +
            # W^K
            [[0]*16]*16 +
            # W^V
            [[0]*16]*16,
            dtype=torch.float))


        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(48))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*16]*16,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(16))

        self.linear1.weight = torch.nn.Parameter(torch.tensor(
            [[1,0,0,0,0,0,0,0]+[0]*8,[0,1,0,0,0,0,0,0]+[0]*8,[0,0,1,0,0,0,0,0]+[0]*8,
            [0,0,0,1,0,0,0,0]+[0]*8,[0,0,0,0,1,0,0,0]+[0]*8,
            [0,0,0,0,0,1,0,0]+[0]*8,[0,0,0,0,0,0,1,0]+[0]*8,[0,0,0,0,0,0,0,1]+[0]*8]+
            [[-1,0,0,0,0,0,0,0]+[0]*8,[0,-1,0,0,0,0,0,0]+[0]*8,[0,0,-1,0,0,0,0,0]+[0]*8,
            [0,0,0,-1,0,0,0,0]+[0]*8,[0,0,0,0,-1,0,0,0]+[0]*8,
            [0,0,0,0,0,-1,0,0]+[0]*8,[0,0,0,0,0,0,-1,0]+[0]*8,[0,0,0,0,0,0,0,-1]+[0]*8]
            , dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(16))

        A = torch.tensor(
            [[-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,-1,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,-1,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,-1,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1]]
            + [[1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,-1,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,-1,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,-1]],
            dtype=torch.float)
        B = torch.tensor([[0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,-1]]+[[0]*16]*14)

        self.linear2.weight = torch.nn.Parameter(torch.add(A,B))
        #We add a small bias in order to avoid zero division during Layernorm 
        self.linear2.bias = torch.nn.Parameter(torch.tensor([-10e-9]*8+[10e-9]*8))    

    def layernorm(self,x):
        y = x
        for i in range(len(y)) :
            mean = 0
            aux = 0
            for sc in y[i][0] :
                mean += sc
                aux += sc**2
            mean /= len(y[i][0])
            aux /= len(y[i][0])
            var = aux-(mean**2)
            for j in range(len(y[i][0])):
                y[i][0][j]=(y[i][0][j]-mean)/math.sqrt(var)  
        return y

    def forward(self, src, src_mask=None, src_key_padding_mask=None, verbose = False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.layernorm(src)
        src1 = self.dropout(self.activation(self.linear1(src)))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src3 = src + src2
        src4 = self.layernorm(src3)
        return src4
    
class MyTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstLayer(),SecondLayer()
        ])
        self.num_layers = len(self.layers)
        self.norm = None

class Model(torch.nn.Module):
    def __init__(self,eta = 10e-6):
        super().__init__()
        self.eta = eta
        self.word_embedding = torch.eye(3, 8)
        self.transformer_encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(16, 1)
        c = -(1/math.sqrt(16/2))*math.log(2**(self.eta)-1) 
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[c,0,0,0,0,0,0,0]+[0]*8], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        x = self.word_embedding[w]
        x2 = torch.Tensor([[0]*16]*len(x))
        for k in range(len(x)):
            aux = -x[k]
            x2[k] = torch.cat((x[k],aux),0)
        y = self.transformer_encoder(x2.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z

length_ls = []
accuracy_ls = []
ce_ls = []
for n in tqdm(range(1,args.length+1)):
    model = Model(10e-3)

    loss = 0
    total = 0
    correct = 0
    for step in range(args.steps):
        w = torch.tensor([alphabet_index['$']] + [alphabet_index[str(random.randrange(2))] for i in range(n)])
        label = check_equality_label(w)
        output = model(w)
        final_output = 1/(1+math.exp(-output))

        if ((final_output) < 0.5 and not(label)) or ((final_output) >= 0.5 and label):
            correct += 1
        #else : 
        #    print(w)
        #    print(final_output)
        #    print(label)
        if label :
            loss -= math.log(final_output,2)
        else :
            loss -= math.log(1-final_output,2)

        total += 1
    #print(f'length={n} ce={loss/total} acc={correct/total}')
    length_ls.append(n)
    ce_ls.append(loss/total/math.log(2))
    accuracy_ls.append(correct/total)

# Figure    
fig, (ax_1, ax_2) = plt.subplots(2,1)

fig.subplots_adjust(hspace=0.5)

# Cross-entropy plot
ax_1.plot(length_ls, ce_ls)
ax_1.set_xlabel('string length n')
ax_1.set_ylabel('cross-entropy (bits)')
ax_1.set(ylim = (0, 1.5), yticks=[0, 0.5, 1, 1.5])

# Accuracy plot
ax_2.plot(length_ls, accuracy_ls, 'r')
ax_2.set_xlabel('string length n')
ax_2.set_ylabel('accuracy')
ax_2.set(ylim = (0, 1.2), yticks=[0, 0.5, 1])

fig.suptitle('EQUALITY')

plt.savefig('equality_exact_layernorm_ce_accuracy.png')