from tqdm import tqdm
from numpy import frombuffer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from NURBSDiff.curve_eval import CurveEval
from cnn import Bottleneck, BasicBlock
from dataloaders import MNISTDataset, CustomDataset
from custom_transforms import im2pc
from pos_encodings import PositionalEmbedding, PositionalEncoding
from loss import VectorizationLoss
import warnings
warnings.filterwarnings("error")


class Net(nn.Module):
    def __init__(self, channels=None,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0,
                 n_controlpoints=4,
                 n_splines=1,
                 n_eval_points=100,
                 batch_size=1):

        super().__init__()
        if channels is None:
            channels = [1, 64, 128, 256]

        self.n_controlpoints = n_controlpoints
        self.n_splines = n_splines
        self.n_eval_points = n_eval_points

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(channels[1], channels[2]),
            BasicBlock(channels[2], channels[3])
            # nn.AdaptiveAvgPool2d((n_controlpoints, n_controlpoints))
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(batch_size, 32, 32,
                                              feat_dim=channels[-1] / 2, normalize=True, device='cuda')

        self.pos_embeder = PositionalEmbedding(n_splines, d_model=channels[-1])

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-1], nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=channels[-1], nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Encoder after transformer
        self.feature_encoder = nn.Sequential(
            nn.Linear(channels[-1], (channels[-1] + n_controlpoints * 2) // 2),
            nn.ReLU(),
            nn.Linear((channels[-1] + n_controlpoints * 2) // 2, n_controlpoints * 2)
        )

        # NURBSDiff curve evaluation module.
        self.curve_weights = torch.ones(n_splines, self.n_controlpoints, 1).cuda()
        self.curve_layer = CurveEval(n_controlpoints, dimension=2, p=3, out_dim=n_eval_points)

    def forward(self, x):
        # print('x_size: ', x.size())

        cnn_out = self.cnn(x)
        # print('cnn_out_size: ', cnn_out.size())

        pe_out = self.pos_encoder(cnn_out).flatten(2, -1).transpose(1, 2)
        # print('pe_out_size: ', pe_out.size())

        memory = self.transformer_encoder(pe_out)
        # print('memory_size: ', memory.size())

        tgt = self.pos_embeder(memory)
        # print('tgt_size: ', tgt.size())

        transformer_out = self.transformer_decoder(tgt, memory)
        # print('transformer_out_size: ', transformer_out.size())

        control_points = self.feature_encoder(transformer_out)
        # print('control_points_size: ', control_points.size())

        out_list = []
        cp_list = []
        for cp in control_points:
            cp = cp.reshape(self.n_splines, self.n_controlpoints, 2)
            cp_list.append(cp)
            cp = torch.cat((cp, self.curve_weights), axis=-1)
            out = self.curve_layer(cp)
            out_list.append(out)

        control_points = torch.stack(cp_list, dim=0)
        out = torch.stack(out_list, dim=0)

        return out, control_points


def val_loop(nett, valloaderr):
    total_val_losss = 0
    with torch.no_grad():
        for val_im, val_target in valloaderr:
            val_im = val_im.cuda()
            val_target = val_target.cuda()

            val_out, val_control_points = nett(val_im)

            val_loss = loss_func(val_out, val_target, val_control_points)

            total_val_losss += val_loss.item()

    return total_val_losss


def plot_output_to_images(test_im, test_target, test_out, test_control_points):
    plots = []
    for b in range(len(test_im)):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(test_im[b][0], extent=[0, 1, 1, 0], cmap='gray')
        plt.scatter(test_target[b, :, 1], test_target[b, :, 0], color='green')
        for j in range(len(test_control_points[0])):
            plt.scatter(test_out[b, j, :, 1], test_out[b, j, :, 0], color='blue')
            plt.scatter(test_control_points[b, j, :, 1], test_control_points[b, j, :, 0], color='red')

        plt.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(2, 0, 1).copy()
        plots.append(torch.from_numpy(plot))
        plt.close()
    return plots


if __name__ == "__main__":
    batch_size = 25
    data_len = 1000
    n_controlpoints = 6
    n_splines = 1
    epochs = 1000
    lr = 1e-4
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    # Create dataloader.
    # dataloader = MNISTDataset('../../datasets_external/mnist', transform=transform, target_transform=im2pc)
    dataset = CustomDataset('./custom_data/random_one_curve_', data_len=data_len, true_targets=True,
                            transform=transform, target_transform=im2pc)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(data_len * 0.85), int(data_len * 0.10),
                                                                           int(data_len * 0.05)])
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size)
    testloader = DataLoader(test_set, batch_size=batch_size)

    # Create network.
    net = Net(channels=None,
              nhead=8,
              num_encoder_layers=6,
              num_decoder_layers=6,
              dim_feedforward=2048,
              dropout=0,
              n_controlpoints=n_controlpoints,
              n_splines=n_splines,
              n_eval_points=100,
              batch_size=batch_size)
    net.cuda()

    # Create logger.
    writer = SummaryWriter('runs/custom_data_vectorization')

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.995)
    loss_func = VectorizationLoss(loss="sinkhorn", p=2, blur=0.01)

    i = 0
    running_loss = 0
    for epoch in tqdm(range(epochs)):
        for im, target in trainloader:
            im = im.cuda()
            target = target.cuda()

            # print('im_size: ', im.size())
            # print('target_size: ', target.size())

            out, control_points = net(im)

            # print('out_size: ', out.size())
            # print('control_points_out_size: ', control_points.size())

            loss = loss_func(out, target, control_points)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Create plot every n training steps.
            if i % 10 == 9:
                writer.add_scalar('training_loss', running_loss / 10, i)
                running_loss = 0

            if i % 100 == 99:
                net.eval()
                total_val_loss = val_loop(net, valloader)
                writer.add_scalar('validation_loss', total_val_loss / len(valloader), i)
                net.train()
            i += 1

        if epoch % 10 == 9:
            with torch.no_grad():
                net.eval()
                test_plots = []
                for batch, (test_im, test_target) in enumerate(testloader):
                    test_im = test_im.cuda()
                    test_target = test_target.cuda()

                    test_out, test_control_points = net(test_im)

                    test_out = test_out.cpu()
                    test_control_points = test_control_points.cpu()
                    test_im = test_im.cpu()
                    test_target = test_target.cpu()

                    test_plot = plot_output_to_images(test_im, test_target, test_out, test_control_points)
                    test_plots += test_plot

                test_grid = make_grid(test_plots, nrow=int(len(test_plots)**0.5))
                writer.add_image('test_images', test_grid, epoch+1)

                train_plots = plot_output_to_images(im.cpu(), target.cpu(), out.detach().cpu(),
                                                    control_points.detach().cpu())
                train_grid = make_grid(train_plots, nrow=int(len(train_plots)**0.5))
                writer.add_image('train_images', train_grid, epoch+1)

                net.train()
