import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy import frombuffer
from torch import from_numpy


def plot_output_to_images(im, target, out, control_points, spline_logits, loss_dict):
    im = im.cpu()
    target = [t.cpu() for t in target]
    out = out.detach().cpu()
    control_points = control_points.detach().cpu()
    spline_logits = spline_logits.detach().cpu()

    class_losses = sum(loss_dict[key] for key in ['loss_ce', 'loss_ce_0', 'loss_ce_1', 'loss_ce_2', 'loss_ce_3',
                                                  'loss_ce_4'])
    distance_losses = sum(loss_dict[key] for key in ['loss_distance', 'loss_distance_0', 'loss_distance_1',
                                                     'loss_distance_2', 'loss_distance_3', 'loss_distance_4'])
    endpoint_losses = sum(loss_dict[key] for key in ['loss_endpoint', 'loss_endpoint_0', 'loss_endpoint_1',
                                                     'loss_endpoint_2', 'loss_endpoint_3', 'loss_endpoint_4'])

    plots = []

    for b in range(len(im)):

        indices = spline_logits[b, :, 0] > spline_logits[b, :, 1]

        fig = plt.figure(figsize=(5, 5))
        img = im[b]
        img -= img.amin(dim=(1, 2), keepdim=True)
        img /= img.amax(dim=(1, 2), keepdim=True)
        plt.imshow(img.permute(1, 2, 0), extent=[0, 1, 1, 0])
        for tt in target[b]:
            plt.scatter(tt[:, 1], tt[:, 0], s=4, color='green')
        for to in out[b, indices]:
            plt.scatter(to[:, 1], to[:, 0], s=4, color='blue')
        for tcp in control_points[b, indices]:
            plt.scatter(tcp[:, 1], tcp[:, 0], color='red')

        plt.annotate(f"Predicted lines: {indices.sum()}", xy=(0.01, 0.05), c='yellow', backgroundcolor='black')
        plt.annotate(f"Class loss: {class_losses[b]:.5f}", xy=(0.01, 0.10), c='red', backgroundcolor='black')
        plt.annotate(f"Distance loss: {distance_losses[b]:.5f}", xy=(0.01, 0.15), c='red', backgroundcolor='black')
        plt.annotate(f"Endpoint loss: {endpoint_losses[b]:.5f}", xy=(0.01, 0.20), c='red', backgroundcolor='black')

        plt.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(2, 0, 1).copy()
        plots.append(from_numpy(plot))
        plt.close()
    return plots