#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def decay_sequence(beta0: float, decay_rate: float, steps: int) -> np.ndarray:
    """Return exponentially decayed sequence: beta_t = beta0 * (1 - decay_rate) ** t for t in [0, steps]."""
    time_steps = np.arange(steps + 1)
    return beta0 * ((1.0 - decay_rate) ** time_steps)


def main() -> None:
    steps = 100

    time_steps = np.arange(steps + 1)
    beta_decay_7_3 = decay_sequence(beta0=7.0, decay_rate=0.03, steps=steps)
    beta_decay_5_5 = decay_sequence(beta0=5.0, decay_rate=0.05, steps=steps)
    beta_decay_7_5 = decay_sequence(beta0=7.0, decay_rate=0.05, steps=steps)
    beta_decay_5_3 = decay_sequence(beta0=5.0, decay_rate=0.03, steps=steps)

    plt.figure(figsize=(8, 4.5))
    plt.plot(time_steps, beta_decay_7_3, label="beta0=7, decay=3%/step")
    plt.plot(time_steps, beta_decay_5_5, label="beta0=5, decay=5%/step")
    plt.plot(time_steps, beta_decay_7_5, label="beta0=7, decay=5%/step")
    plt.plot(time_steps, beta_decay_5_3, label="beta0=5, decay=3%/step")

    plt.xlabel("Step")
    plt.ylabel("Beta")
    plt.title("Per-step Exponential Decay of Beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("beta_decay.png")


if __name__ == "__main__":
    main()
