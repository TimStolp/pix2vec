from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from NURBSDiff.curve_eval import CurveEval
from pytorch3d.loss import chamfer_distance
from cnn import Bottleneck
from dataloader import MNISTDataset, CustomDataset
from transforms import im2pc, MyRotationTransform
from positional_encodings import PositionalEmbedding, PositionalEncoding


class Net(nn.Module):
    def __init__(self, channels=None,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1, n_controlpoints=4, batch_size=1):

        super().__init__()
        if channels is None:
            channels = [1, 64, 128, 256]

        self.n_controlpoints = n_controlpoints

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(channels[1], channels[2]),
            Bottleneck(channels[2], channels[3])
            # nn.AdaptiveAvgPool2d((n_controlpoints, n_controlpoints))
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(batch_size, 7, 7,
                                              feat_dim=channels[-1]/2, normalize=True, device='cuda')

        self.pos_embeder = PositionalEmbedding(n_controlpoints, d_model=channels[-1])

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
            nn.Linear(channels[-1], channels[-1]//2),
            nn.ReLU(),
            nn.Linear(channels[-1]//2, 2)
        )

        # NURBSDiff curve evaluation module.
        self.curve_layer = CurveEval(n_controlpoints, dimension=2, p=3, out_dim=100)

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

        # control_points = control_points[0].reshape(2, self.n_controlpoints, 2)
        # print('reshapes_control_points_size: ', control_points.size())

        # NURBS weights all set to 1 (which makes it a standard b-spline).
        self.curve_weights = torch.ones(1, self.n_controlpoints, 1).cuda()
        control_points = torch.cat((control_points, self.curve_weights), axis=-1)

        # Get point cloud from control points.
        out = self.curve_layer(control_points).reshape(1, 100, 2)
        # print('out_size: ', out.size())

        return out, control_points


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    # MyRotationTransform(angle=90),
                                    transforms.Normalize(0.5, 0.5)])

    # dataloader = MNISTDataset('../../datasets_external/mnist', transform=transform, target_transform=im2pc)
    dataloader = CustomDataset('./custom_data', true_targets=True, transform=transform, target_transform=im2pc)
    net = Net(channels=None,
              nhead=8,
              num_encoder_layers=6,
              num_decoder_layers=6,
              dim_feedforward=2048,
              dropout=0.2,
              n_controlpoints=4,
              batch_size=1)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    pbar = tqdm(range(10001))
    pc_losses = []
    cp_losses = []
    total_losses = []

    for i in pbar:
        optimizer.zero_grad()

        im, target = dataloader['one_curve']
        im = im.cuda()
        target = target.cuda()

        # print('im_size: ', im.size())
        # print('target_size: ', target.size())

        out, control_points = net(im)

        # print('out_size: ', out.size())

        loss, _ = chamfer_distance(out, target, point_reduction='mean')

        loss.backward()

        optimizer.step()
        scheduler.step()

        pc_losses.append(loss.item())

        # Create plot every n training steps.
        if i % 1000 == 0:
            # print("Out")
            # print(out)
            # print("Target")
            # print(target)

            net.eval()

            out, control_points = net(im)

            out = out.cpu().detach()
            control_points = control_points.cpu().detach()
            target = target.cpu()
            im = im.cpu()

            print('---')
            print(control_points)

            plt.figure()
            plt.imshow(im[0][0], extent=[0, 1, 1, 0], cmap='gray')
            plt.scatter(target[0, :, 1], target[0, :, 0], color='green')
            plt.scatter(out[0, 0:100, 1], out[0, 0:100, 0], color='blue')
            # plt.scatter(out[0, 100:200, 1], out[0, 100:200, 0], color='yellow')

            for cp in control_points:
                plt.scatter(cp[:, 1], cp[:, 0])
            plt.savefig(f'./outputImages/predicted_spline_{i}.png')
            plt.close()

            net.train()

    plt.figure()
    plt.plot(pc_losses)
    plt.savefig('./outputImages/training_pointcloud_loss.png')
    plt.close()
