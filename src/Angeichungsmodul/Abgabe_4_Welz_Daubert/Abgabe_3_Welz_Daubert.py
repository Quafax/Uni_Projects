import numpy as np
import matplotlib.pyplot as plt

def euler(t_end, y_0, f, h):
    ts = np.arange(0, t_end + h/2, h)
    ys = np.empty_like(ts, dtype=float)
    ys[0] = y_0

    for i in range(len(ts) - 1):
        ys[i + 1] = ys[i] + h * f(ys[i], ts[i])

    return ts, ys

def f(y, t):
    return -0.5 * y

if __name__ == "__main__":
    y0 = 1.0

    ts1, y1 = euler(10, y0, f, 0.1)
    ts2, y2 = euler(10, y0, f, 0.5)
    ts3, y3 = euler(10, y0, f, 1)

    plt.figure()
    plt.plot(ts1, y1, label="h=0.1")
    plt.plot(ts2, y2, label="h=0.5")
    plt.plot(ts3, y3, label="h=1")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Euler-Verfahren")
    plt.grid(True)
    plt.legend()
    plt.show()