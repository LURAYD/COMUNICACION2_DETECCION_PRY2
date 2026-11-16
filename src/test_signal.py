"""Demostración manual: python -m src.test_signal."""
import numpy as np

from src.signals import bpsk_modulation


def main():
    import matplotlib.pyplot as plt

    bits = [0, 1, 0, 1, 1, 0]
    symbols = bpsk_modulation(bits)
    power = np.mean(np.abs(symbols)**2)
    print("Bits:", bits)
    print("Símbolos BPSK:", symbols)
    print("Energía media:", power)
    assert np.isclose(power, 1.0), "La energía media BPSK debe ser 1."
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(symbols.real, symbols.imag)
    ax.set(xlabel="I", ylabel="Q", title="Constelación BPSK", aspect="equal")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
