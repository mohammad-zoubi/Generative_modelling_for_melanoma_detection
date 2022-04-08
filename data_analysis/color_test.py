import seaborn as sns; sns.set_theme()
from numpy import arange
import matplotlib.pyplot as plt


x = arange(25).reshape(5, 5)
cmap = sns.diverging_palette(7, 333, l=60, s=35,center='dark',as_cmap=True)
ax = sns.heatmap(x, cmap=cmap)

fig = plt.figure(figsize= (12,12))
fig = sns.heatmap(x, cmap=cmap)
fig = fig.get_figure()
fig.savefig('test.jpg')