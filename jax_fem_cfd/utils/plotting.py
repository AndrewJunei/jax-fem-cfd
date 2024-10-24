import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
plt.rcParams['font.family'] = 'Arial'

def show_plots():
    for i in plt.get_fignums(): # iterate over all active plot numbers
        fig = plt.figure(i)
        # get the window manager for the figure
        manager = fig.canvas.manager
        window = manager.window
        # get window flags to stay on top and active
        window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
        window.show()  
        window.activateWindow() # activate the window to bring it to front
    plt.show()

def get_2d_arrays(x, y, u, v, nnx, nny):
    x_unique = np.linspace(x.min(), x.max(), nnx)
    y_unique = np.linspace(y.min(), y.max(), nny)
    x2d, y2d = np.meshgrid(x_unique, y_unique)

    u2d = np.array(u.reshape(nny, nnx))
    v2d = np.array(v.reshape(nny, nnx))

    return x2d, y2d, u2d, v2d 

def get_2d_tri(x, y):
    return Triangulation(x, y)

def plot_contour(tri, f, x, y, u, v, nnx, nny, title, figsize=None, c_shrink=1.0, quiv=False,
                 xlbl=None, ylbl=None):
    plt.figure(figsize=figsize)

    plt.tricontourf(tri, f, levels=40, cmap='jet')
    plt.colorbar(shrink=c_shrink)

    if quiv:
        num = min(nnx, nny, 50)

        xi = np.linspace(x.min(), x.max(), num)
        yi = np.linspace(y.min(), y.max(), num)
        X, Y = np.meshgrid(xi, yi)

        U = griddata((x, y), u, (X, Y), method='linear')
        V = griddata((x, y), v, (X, Y), method='linear')
        plt.quiver(X, Y, U, V)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title, fontsize=12)
    plt.xlabel(xlbl, fontsize=12)
    plt.ylabel(ylbl, fontsize=12)

def plot_image(f, x, y, u, v, nnx, nny, title, figsize=None, c_shrink=1.0, quiv=False,
                 xlbl=None, ylbl=None):
    plt.figure(figsize=figsize)

    f = np.flip(f.reshape(nny, nnx), axis=0)

    plt.imshow(f, cmap='jet', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.colorbar(shrink=c_shrink)

    if quiv:
        num = min(nnx, nny, 50)

        xi = np.linspace(x.min(), x.max(), num)
        yi = np.linspace(y.min(), y.max(), num)
        X, Y = np.meshgrid(xi, yi)

        U = griddata((x, y), u, (X, Y), method='linear')
        V = griddata((x, y), v, (X, Y), method='linear')
        plt.quiver(X, Y, U, V)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title, fontsize=12)
    plt.xlabel(xlbl, fontsize=12)
    plt.ylabel(ylbl, fontsize=12)
    plt.xticks([])
    plt.yticks([])

def plot_surface(x, y, f, title, figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x, y, f, cmap='jet', edgecolor='none')
    fig.colorbar(surf, shrink=0.8)
    ax.set_title(title, pad=0)
    ax.xaxis.set_pane_color('white')  
    ax.yaxis.set_pane_color('white') 
    ax.zaxis.set_pane_color('white')
    plt.tight_layout()

def plot_streamlines(x2d, y2d, u2d, v2d, title, figsize=None):
    plt.figure(figsize=figsize)

    plt.streamplot(x2d, y2d, u2d, v2d, density=1.5, linewidth=0.5, arrowstyle='-', color='black')
    plt.xlim(x2d.min(), x2d.max())
    plt.ylim(y2d.min(), y2d.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title, fontsize=12)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)

def compare_double(xref1, ref1, reflbl1, x1, sol1, lbl1, xref2, ref2, reflbl2, x2, sol2, lbl2,
                   title1, title2, xlbl1, xlbl2):
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(xref1, ref1, label=reflbl1, color='red', marker='s')
    axs[0].plot(x1, sol1, label=lbl1, color='blue', linewidth=1.5)
    axs[0].set_xlabel(xlbl1, fontsize=12)
    axs[0].set_ylabel('u', fontsize=12)
    axs[0].set_title(title1, fontsize=12)

    axs[1].scatter(xref2, ref2, label=reflbl2, color='red', marker='s')
    axs[1].plot(x2, sol2, label=lbl2, color='blue', linewidth=1.5)
    axs[1].set_xlabel(xlbl2, fontsize=12)
    axs[1].set_ylabel('v', fontsize=12)
    axs[1].set_title(title2, fontsize=12)

    for ax in axs:
        ax.legend(fontsize=12)
        ax.grid(True, color='#DDDDDD')
        ax.set_axisbelow(True)

    plt.tight_layout()

def plot_solve_iters(momentum_iters, pressure_iters, steps):
    plt.figure()
    plt.plot(steps, momentum_iters, label='Momentum: BiCGSTAB', color='red', marker='o')
    plt.plot(steps, pressure_iters, label='Pressure: CG', color='black', marker='s')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Iterations', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)