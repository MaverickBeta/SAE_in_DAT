import numpy as np
import matplotlib.pyplot as plt

data = np.load("acts_mean.npz")
A = data['A_clean_python']
D = data['D_clean_sealion']
E = data['E_attack_sl2py']

print(A.shape)
N = len(A)
N1 = N // 3
N3 = N // 3
N2 = N - N1 - N3

diff = A - D
idx_diff = np.argsort(diff)[::-1]

left_idx = idx_diff[:N1]
right_idx = idx_diff[-N3:]
mid_idx = idx_diff[N1:-N3]

# left sorted by A descending
left_idx = left_idx[np.argsort(A[left_idx])[::-1]]

# right sorted by D ascending (so D decreases from right to left)
right_idx = right_idx[np.argsort(D[right_idx])]

final_idx = np.concatenate([left_idx, mid_idx, right_idx])

plt.figure(figsize=(14, 6))
x = np.arange(N)

# Plot E first so it's in the background, or A and D first?
plt.plot(x, A[final_idx], color='blue', alpha=0.9, label='A_clean_python', linewidth=1.5)
plt.plot(x, D[final_idx], color='red', alpha=0.9, label='D_clean_sealion', linewidth=1.5)
plt.plot(x, E[final_idx], color='green', alpha=0.8, label='E_attack_sl2py', linewidth=1.5, linestyle='--')

plt.axvline(x=N1, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=N1+N2, color='gray', linestyle=':', alpha=0.5)

plt.title("Latent Variable Activation Analysis")
plt.legend()
plt.savefig("test_plot_v2.png")
print("Done")
