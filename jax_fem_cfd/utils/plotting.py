import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.font_manager import fontManager

available_fonts = set(f.name for f in fontManager.ttflist)

for font in ['Arial', 'Liberation Sans', 'sans-serif']:
    if font in available_fonts:
        plt.rcParams['font.family'] = font
        break

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

def plot_contour(tri, f, x, y, u, v, nnx, nny, title=None, figsize=None, c_shrink=1.0, quiv=False,
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

def plot_surface(x, y, f, title=None, figsize=None):
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

def save_animation(phi_list, path, dt, nnx, nny, duration_seconds, init_time_seconds, constant_lim):
    fig, ax = plt.subplots(figsize=(10, 8))

    if constant_lim:
        im = ax.imshow(np.zeros((nny, nnx)), cmap='jet', vmin=0, vmax=1)
        # im = ax.imshow(np.zeros((nny, nnx)), cmap='Purples', vmin=0, vmax=1)
    else:
        im = ax.imshow(np.zeros((nny, nnx)), cmap='jet')
        # im = ax.imshow(np.zeros((nny, nnx)), cmap='Purples')

    _ = fig.colorbar(im, ax=ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("", fontsize=12)
   
    max_fps = 50  # reasonable upper limit

    # Calculate fps needed for duration
    ideal_fps = len(phi_list) / duration_seconds
    fps = min(ideal_fps, max_fps)  # cap at max_fps
    
    # Calculate frame step based on final fps
    step = max(1, round(len(phi_list) / (duration_seconds * fps)))
    frames_to_use = phi_list[::step]

    # Show initial condition longer
    initial_frames = round(init_time_seconds * fps)  # number of frames to show initial condition
    frames_to_use = [frames_to_use[0]] * initial_frames + list(frames_to_use)
    n_frames = len(frames_to_use)
   
    def animate(frame):
        if frame < initial_frames:
            f = frames_to_use[0]
            actual_time = 0
        else:
            f = frames_to_use[frame]
            actual_time = ((frame - initial_frames) * step * dt)

        f = np.flip(f.reshape(nny, nnx), axis=0)
        im.set_array(f)
        if not constant_lim:
            im.set_clim(vmin=np.min(f), vmax=np.max(f))
        title.set_text(f'Concentration at time {actual_time:.2f}')
        return [im]

    print(f"Number of frames: {n_frames}")
    print(f"FPS: {fps}")
    print(f"Expected duration: {n_frames/fps} seconds")

    anim = FuncAnimation(fig, animate, frames=n_frames, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(f'{path}.gif', writer=writer)
    plt.close()