import torch
import torch.nn as nn
from NURBSDiff.curve_eval import CurveEval
from pos_encodings import PositionalEmbedding, PositionalEncoding
from resnet import resnet18
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer


class Net(nn.Module):
    def __init__(self, channels=None,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.0,
                 n_controlpoints=4,
                 n_splines=1,
                 n_eval_points=100,
                 batch_size=1):

        super().__init__()
        if channels is None:
            # channels = [1, 64, 128, 256]
            channels = [3, 64, 256, 512]

        self.n_controlpoints = n_controlpoints
        self.n_splines = n_splines
        self.n_eval_points = n_eval_points

        # Resnet18 CNN
        self.cnn = resnet18()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(batch_size, 16, 16,
                                              feat_dim=channels[-1] / 2, normalize=True)

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
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], n_controlpoints * 2)
        )

        self.probability_encoder = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], 2)
        )

        # NURBSDiff curve evaluation module.
        self.register_buffer('curve_weights', torch.ones(n_splines, self.n_controlpoints, 1))
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

        spline_logits = self.probability_encoder(transformer_out)
        # print('spline_logits_size: ', spline_logits.size())

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
        # print('out_size: ', out.size())

        return out, control_points, spline_logits
