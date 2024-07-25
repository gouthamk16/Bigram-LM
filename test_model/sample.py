# nlls = torch.zeros(5)
# num = 5
# for i in range(num):
#     # iith bigram
#     x = xs[i].item()
#     y = ys[i].item()
#     print('------------------------------------')
#     print(f"Bigram example {i+1} : {itos[x]}{itos[y]} | Indexes: {x},{y}")
#     print(f"Input to the neural net: {x}")
#     print(f"Output probabilities from the neural net: {probs[i]}")
#     print(f"Actual next character (label): {y}")
#     p = probs[i, y]
#     print(f"Prob assigned by our model to the correct label: {p.item():.4f}")
#     logp = torch.log(p)
#     print(f"Log likelihood: {logp.item():.4f}")
#     nll = -logp
#     print(f"Negative log likelihood: {nll.item():.4f}")
#     nlls[i] = nll

# print("*****************************************")
# print(f"Average negative log likelihood for {num} samples (loss): {nlls.mean().item():.4f}")
import torch


name = [13, 12]
C  = torch.randn((27, 10))
emb = C[name]
print(emb)
