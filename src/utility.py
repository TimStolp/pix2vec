import matplotlib.pyplot as plt
from numpy import frombuffer
from torch import from_numpy


def plot_output_to_images(test_im, test_target, test_out, test_control_points):
    plots = []
    for b in range(len(test_im)):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(test_im[b][0], extent=[0, 1, 1, 0], cmap='gray')
        for tt in test_target[b]:
            plt.scatter(tt[:, 1], tt[:, 0], color='green')
        for to in test_out[b]:
            plt.scatter(to[:, 1], to[:, 0], color='blue')
        for tcp in test_control_points[b]:
            plt.scatter(tcp[:, 1], tcp[:, 0], color='red')

        plt.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(2, 0, 1).copy()
        plots.append(from_numpy(plot))
        plt.close()
    return plots