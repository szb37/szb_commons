import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'arial'})
plt.rcParams['svg.fonttype'] = 'none'  # Ensure fonts are embedded
plt.rcParams['text.usetex'] = False  # Use TeX to handle text (embeds fonts)
title_fontdict = {'fontsize': 20, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 14, 'fontweight': 'bold'}
ticklabel_fontsize = 14

save_PNG = True
save_SVG = False
