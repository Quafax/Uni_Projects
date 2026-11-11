#imports
import numpy as np
import matplotlib.pyplot as plt

#euler approximation
def euler(t_end, y_0, f, h):
    """
    t_end is maximal steps, y_0 is anfangsbedingung, f is function, h is schrittweite
    """
    #array for t
    ts = np.arange(0, t_end + h/2, h)
    #array for ys
    ys = np.empty_like(ts, dtype=float)
    #fill in anfangsbedingung
    ys[0] = y_0

    #alle ts durchgehen und mit euler aproximation formel nächstes y ausrechnen
    for i in range(len(ts) - 1):
        ys[i + 1] = ys[i] + h * f(ys[i], ts[i])
    #returnen
    return ts, ys

#function definieren
def f(y, t):
    return -0.5 * y

#nur ausführen wenn direkt ausgeführt
if __name__ == "__main__":
    #anfangsbedingung
    y0 = 1.0

    #euler für verschiedene hs
    ts1, y1 = euler(10, y0, f, 0.1)
    ts2, y2 = euler(10, y0, f, 0.5)
    ts3, y3 = euler(10, y0, f, 1)

    #plotte das alles
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