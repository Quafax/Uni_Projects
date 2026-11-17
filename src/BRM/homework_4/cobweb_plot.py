import numpy as np
import matplotlib.pyplot as plt


def logistic_func(r, z):
    return r * z * (1 - z)

def two_step_func(r, z):
    #make the logistic but do two steps
    z_1 = logistic_func(r, z)
    z_2 = logistic_func(r, z_1)
    return z_2

def cobweb_diagram(r, z_0, xs, ax, steps=30):
    # show function and diagonal
    ax.plot(xs, logistic_func(r, xs), label="f(z) = r * z * (1 - z)")
    ax.plot(xs, xs, "k--", label="y = z")

    #start point
    z = z_0
    y = 0

    for _ in range(steps):
        z_next = logistic_func(r, z)

        #go perpendicular to original func
        ax.plot([z, z], [y, z_next], color="crimson", linewidth=1)

        #horizontal to diag
        ax.plot([z, z_next], [z_next, z_next], color="crimson", linewidth=1)

        #nexxt startpoint
        z = z_next
        y = z_next

    #set the basics
    ax.set_title(f"r = {r}, Starting point = {z_0}")
    ax.set_xlabel(r"$z_n$")
    ax.set_ylabel(r"$z_{n+1}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if r == 0.5:
        ax.set_xlim(0, 0.12)
        ax.set_ylim(0, 0.12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    #if monoton sstable or oszillierend
    beschriftungen = {
        0.5: "Monoton stabil",
        1.5: "Monoton stabil",
        2.5: "Oszillierend stabil",
        3.3: "Oszillierend stabil um 2 Punkte"}

#show what it is
    ax.text( 1, 1, beschriftungen.get(r, ""), transform=ax.transAxes, ha="right", va="top")

if __name__ == "__main__":
    z_0 = 0.1
    r = [0.5, 1.5, 2.5, 3.3]
    xs = np.linspace(0, 1, 500)

    #show 4 diags
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    #generate the plots
    for r_value, ax in zip(r, axes.flat):
        cobweb_diagram(r_value, z_0, xs, ax)

    #show plots
    plt.tight_layout()
    plt.show()