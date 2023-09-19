import numpy as np
import matplotlib.pyplot as plt

N = 100
theta = np.pi * (1) #change the parameter theta, phi_1, Phi_2
phi_1 = 0
phi_2 = 0
alpha = 1/np.sqrt(2)
beta = 1j/np.sqrt(2)

n = 2 * N + 3
mid = N + 2

C = np.array([[np.cos(theta), np.exp(1j * phi_1) * np.sin(theta)],
              [np.exp(1j * phi_2) * np.sin(theta), - np.exp(1j * (phi_1 + phi_2)) * np.cos(theta)]])

A = np.zeros((2,n),complex)
B = np.zeros((2,n),complex)

B[0, mid] = alpha
B[1, mid] = beta

# Amplitude Calculation
for k in range(1, N + 1):
    for m in range(2, 2 * N + 2):
        A[0, m] = C[0, 0] * B[0, m - 1] + C[0, 1] * B[1, m - 1]
        A[1, m] = C[1, 0] * B[0, m + 1] + C[1, 1] * B[1, m + 1]
    B = A

P = np.zeros(n)
for m in range(n):
    P[m] = np.abs(B[0, m]) ** 2 + np.abs(B[1, m]) ** 2
    xm = -((n + 1) / 2) + m

pt = np.transpose(P)
t = np.linspace(1, 2 * N + 3, 2 * N + 3)
plt.plot(t, pt)
plt.show()