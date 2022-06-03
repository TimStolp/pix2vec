import matplotlib.pyplot as plt
from numpy import frombuffer
from torch import from_numpy


def plot_output_to_images(im, target, out, control_points, spline_logits):
    plots = []

    for b in range(len(im)):

        indices = spline_logits[b, :, 0] > spline_logits[b, :, 1]

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(im[b][0], extent=[0, 1, 1, 0], cmap='gray')
        for tt in target[b]:
            plt.scatter(tt[:, 1], tt[:, 0], color='green')
        for to in out[b, indices]:
            plt.scatter(to[:, 1], to[:, 0], color='blue')
        for tcp in control_points[b, indices]:
            plt.scatter(tcp[:, 1], tcp[:, 0], color='red')

        plt.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(2, 0, 1).copy()
        plots.append(from_numpy(plot))
        plt.close()
    return plots