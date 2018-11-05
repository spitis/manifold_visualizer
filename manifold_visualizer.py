import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_mxn(m, n, images):
  """Plots mxn MNIST images"""
  images = images.reshape((m*n,28,28))
  fig = plt.figure(1, (n, m))
  grid = ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=(m, n),  # creates grid of axes
                   axes_pad=0.1,  # pad between axes in inch.
                   share_all=True,
                   )
  
  
  grid.axes_llc.set_xticks([])
  grid.axes_llc.set_yticks([])
  
  for i in range(m*n):
    grid[i].set_axis_off()
    grid[i].imshow(images[i], cmap = cm.Greys)  # The AxesGrid object work as a list of axes.

  plt.show()


def animate_mnist_manifold(manifold):
  manifold = manifold.reshape([-1, 28, 28])
  fig, ax = plt.subplots()
  ax.set_axis_off()
  ax = ax.imshow(manifold[0], cmap = cm.Greys)
  
  def animate(i):
    ax.set_data(manifold[i])
    return (ax,)

  # call the animator. blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, init_func=lambda: animate(0),
                                 frames=len(manifold), interval=5, blit=True)
  
  return anim

def draw_mxn_mnist_manifolds(manifold_generator, m, n, portion_to_draw=0.5):
  res = []
  for i in range(m):
    r = []
    while len(r) != n * 300:
      r = manifold_generator(10*n)
      
    res.append(r[np.arange(0, n*int(portion_to_draw*300), int(portion_to_draw*300))])
  res = np.array(res)
  plot_mxn(m, n, res)
