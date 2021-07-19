import matplotlib.pyplot as plt

def plot_side_by_side(l_img, r_img, l_desc='Original Image', r_desc='New Image', l_cmap=None, r_cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(l_img, cmap=l_cmap)
    ax1.set_title(l_desc, fontsize=50)
    ax2.imshow(r_img, cmap=r_cmap)
    ax2.set_title(r_desc, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
