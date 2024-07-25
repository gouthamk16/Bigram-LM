import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


words = open('data/names.txt', 'r').read().splitlines()
# print(words[:10])
# print(len(words))

# bigram lang model - given a character, we predict the next character in the sequence

# b = {}
# for w in words:
#     chs = ['<S>'] + list(w) + ['<E>']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         bigram = (ch1, ch2)
#         b[bigram] = b.get(bigram, 0) + 1

# print(sorted(b.items(), key = lambda kv : kv[1]))

# Creating a tensor to represent the bigram count
N = torch.zeros((27, 27), dtype=torch.int32)

# Mapping each character to a number
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

# Creating a mapping from number to character
itos = {i:s for s, i in stoi.items()}

# print(stoi)

# For each bigram, map each bigram to a number, store the count of the bigram in the tensor
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

## Counts help us to understand how often a bigram starts with a particular word

# Visualizing the tensor
plt.figure(figsize=(30, 30))
plt.imshow(N, cmap='Reds')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
# plt.show()

# Sampling from the count tensor using a probability distribution
g = torch.Generator().manual_seed(2147483647)

# p = N[0].float() # Probabilty of each count in the first row of the tensor
# p = p / p.sum()
# print(p, sum(p))
# Torch.multinomial - sampling from the distribution
# p = torch.rand(3, generator=g)
# p = p / p.sum() # Torch tensor of prob distrbutions

# torch.multinomial (p, num_samples=20, replacement=True, generator=g)
# Input = {[0.68, 0.32, 0.09]} -> {index 0, index 1 , index 2}
# The output will contain 60% 0's, 32% 1's and 0.9% 2's
# Output = {[1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]}

# Creating a tensor to store the probability distributions
P = (N+1).float() # Smoothening the dataset by adding fake value (adding 1)
# Copies the counts and performs an element wise division - check broadcasting semantics for more
P /= P.sum(1, keepdim=True) # Keepdim keeps the input dimensions as it is. is the input is 27*27, the shape of the output sum tensor will also be 2dimensional. 

for i in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    # print(''.join(out))

# Determining the quality of our model using training loss
# Maximum likelihood estimation - product of all the probabilites
# We are going to measure the probability of the training data estimated by the model - hence should be as high as possible
# We consider the log likelihood since multiplying all the probabilities will lead to a very small uninterpretable value
# For a loss funtion, low is good. Hence we take the inverse log likelihood

log_likelihood = 0
n = 0 # For calculating the normalized log likelihood
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logProb = torch.log(prob)
        log_likelihood += logProb
        n += 1
        # print(f"{ch1}{ch2}: {prob:.4f}, Log Prob: {logProb:.4f}") # Probability of each bigram for the first 3 entries in the ds

# print(f"Log Likelihood: {log_likelihood:.4f}")
nll = -log_likelihood
# print(f"NLL: {nll:.4f}")
# Normalized log likelihood
# print(f"Normalized NLL: {nll/n:.4f}")

# Goal - to maximize log likelihood, i.e., minimize avg negative log likelihood

####### THE NEURAL NET ########

# Outputs the prob distribution given an input bigram

# Create the training set of all the bigrams (x, y)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs) # You can also use torch.Tensor - returns in float32
ys = torch.tensor(ys)
num = xs.nelement()
print("Number of samples: ",num)

# Initializing the weight vectors
W = torch.randn((27, 27), generator=g, requires_grad=True)

# Epochs and Learning Rate - Neural network parameters
epochs = 500
lr = 10 # Simple dataset, no possibility to overdo smthng, think big
lambda_reg = 0.1 # regularization param

for epoch in range(epochs):
    # Forward Pass
    xenc = F.one_hot(xs, num_classes=27).float() # 27 classes classes in xs will have value 1
    logits = xenc @ W # tensor multiplication -> log counts
    counts = logits.exp() 
    probs = counts / counts.sum(1, keepdims=True) # -> Softmax activation function (shape = (num x 27))
    loss = -probs[torch.arange(num), ys].log().mean() + lambda_reg*(W ** 2).mean() # Avg NLL with regularization
    print(f"Epoch: {epoch+1} | NLL Loss: {loss}") 

    # Backprop
    W.grad = None
    loss.backward()
    W.data += -lr * W.grad

# Sampling from our model
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break;
    print(''.join[out])