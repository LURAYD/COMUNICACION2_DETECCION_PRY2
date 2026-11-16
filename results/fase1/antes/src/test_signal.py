import numpy as np
import matplotlib.pyplot as plt

from signals import bpsk_modulation

bits = [0, 1, 0, 1, 1, 0]  # Example input bits

symbol = bpsk_modulation(bits)
print("Input bits:", bits)
print("BPSK symbols:", symbol)

power = np.mean(np.abs(symbol)** 2)
print("Power of the BPSK signal:", power)

assert np.isclose(power, 1.0), "Power of the BPSK signal is not close to 0."

plt.figure(figsize=(10, 6))
plt.scatter(symbol.real, symbol.imag, color='blue', marker='o')

plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginaria")
plt.title("Constelación BPSK")
plt.grid(True)
plt.axis("equal")

plt.show()

