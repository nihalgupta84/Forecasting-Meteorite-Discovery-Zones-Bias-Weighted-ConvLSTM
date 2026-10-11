"""
model.py
────────
ConvLSTM-based spatiotemporal model for meteorite density map prediction.

Architecture:
  Input  → ConvLSTM Encoder (2 layers) → Conv2D Decoder → Output density map

Reference: Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting." NeurIPS 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ── ConvLSTM Cell ──────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.

    Performs LSTM gating with 2D convolutions instead of matrix multiplications.
    Maintains spatial structure throughout the sequence.

    Args:
        in_channels  : channels in input tensor x_t
        hidden_channels : channels in hidden state h_t and cell state c_t
        kernel_size  : spatial kernel size (odd integer, e.g. 3)
    """

    def __init__(self,
                 in_channels:     int,
                 hidden_channels: int,
                 kernel_size:     int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2   # "same" padding

        # Single conv for all 4 gates (i, f, o, g) — efficient
        self.conv = nn.Conv2d(
            in_channels  = in_channels + hidden_channels,
            out_channels = 4 * hidden_channels,
            kernel_size  = kernel_size,
            padding      = pad,
            bias         = True,
        )

    def forward(self,
                x_t: torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor):
        """
        Args:
            x_t    : (B, C_in, H, W)
            h_prev : (B, C_hidden, H, W)
            c_prev : (B, C_hidden, H, W)
        Returns:
            h_t, c_t : both (B, C_hidden, H, W)
        """
        combined = torch.cat([x_t, h_prev], dim=1)  # (B, C_in+C_hidden, H, W)
        gates    = self.conv(combined)               # (B, 4*C_hidden, H, W)

        i, f, o, g = torch.chunk(gates, 4, dim=1)   # each (B, C_hidden, H, W)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def init_hidden(self, batch_size: int, height: int, width: int,
                    device: torch.device):
        """Return zero initial hidden and cell states."""
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width,
                        device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width,
                        device=device),
        )


# ── ConvLSTM Layer ─────────────────────────────────────────────────────────────

class ConvLSTMLayer(nn.Module):
    """
    A full ConvLSTM layer that processes a sequence of T frames.

    Args:
        in_channels     : input channel count
        hidden_channels : hidden state channel count
        kernel_size     : conv kernel size
        return_all_seq  : if True, return all T hidden states; else only last
    """

    def __init__(self,
                 in_channels:     int,
                 hidden_channels: int,
                 kernel_size:     int  = 3,
                 return_all_seq:  bool = False):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.return_all_seq = return_all_seq

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, T, C, H, W)
        Returns:
            out : (B, T, C_hidden, H, W) if return_all_seq
                  (B, C_hidden, H, W)    if not return_all_seq
        """
        B, T, C, H, W = x.shape
        device = x.device
        h, c = self.cell.init_hidden(B, H, W, device)

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            if self.return_all_seq:
                outputs.append(h)

        if self.return_all_seq:
            return torch.stack(outputs, dim=1)   # (B, T, C_hidden, H, W)
        else:
            return h                              # (B, C_hidden, H, W)


# ── Full Model ─────────────────────────────────────────────────────────────────

class MeteoriteConvLSTM(nn.Module):
    """
    Full spatiotemporal model for meteorite density map prediction.

    Encoder: N stacked ConvLSTM layers (N = len(hidden_channels))
    Decoder: 3-layer Conv2D with ReLU activations + sigmoid output

    Input  : (B, T, 1, H, W)  — sequence of T cumulative density maps
    Output : (B, 1, H, W)     — predicted next-decade incremental map
    """

    def __init__(self,
                 hidden_channels: list = None,
                 kernel_size:     int  = None):
        super().__init__()

        hidden_channels = hidden_channels or config.CONVLSTM_HIDDEN
        kernel_size     = kernel_size     or config.CONVLSTM_KERNEL

        # ── Encoder: variable-depth stacked ConvLSTM layers ──────────────────
        # All layers except the last pass their full sequence forward;
        # the last layer returns only the final hidden state.
        layers = []
        for i, hid in enumerate(hidden_channels):
            in_ch = 1 if i == 0 else hidden_channels[i - 1]
            is_last = (i == len(hidden_channels) - 1)
            layers.append(ConvLSTMLayer(
                in_channels     = in_ch,
                hidden_channels = hid,
                kernel_size     = kernel_size,
                return_all_seq  = not is_last,
            ))
        self.encoder = nn.ModuleList(layers)

        # ── Batch Norm after encoder ──────────────────────────────────────────
        self.bn = nn.BatchNorm2d(hidden_channels[-1])

        # ── Decoder: Conv2D layers ────────────────────────────────────────────
        decoder_ch = config.DECODER_CHANNELS   # [64, 32, 1]
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], decoder_ch[0],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(decoder_ch[0]),

            nn.Conv2d(decoder_ch[0], decoder_ch[1],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(decoder_ch[1], decoder_ch[2],
                      kernel_size=1),          # 1×1 conv for channel reduction
            nn.Sigmoid(),                      # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 1, H, W)
        Returns:
            pred : (B, 1, H, W)
        """
        out = x
        for layer in self.encoder:
            out = layer(out)   # (B, T, hid, H, W) for all but last
                               # (B, hid, H, W) for last layer
        out = self.bn(out)
        return self.decoder(out)   # (B, 1, H, W)


# ── CBAM Attention Gate ────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).
    Learns to re-weight channels by global average + max pooling.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.flatten(2).mean(dim=2)         # (B, C) global avg
        mx  = x.flatten(2).max(dim=2).values   # (B, C) global max
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * att[:, :, None, None]


class SpatialAttention(nn.Module):
    """
    Spatial attention from CBAM (Woo et al., ECCV 2018).
    Highlights important spatial locations across all channels.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)      # (B,1,H,W)
        mx  = x.max(dim=1, keepdim=True).values
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Full CBAM: channel attention → spatial attention."""
    def __init__(self, channels: int, reduction: int = 8, spatial_k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── Attention-ConvLSTM ─────────────────────────────────────────────────────────

class AttentionConvLSTM(nn.Module):
    """
    MeteoriteConvLSTM augmented with a CBAM attention gate between the
    encoder and decoder.

    The attention module re-weights the encoder's final hidden state both
    channel-wise and spatially before decoding, allowing the model to focus
    on the most informative feature maps and grid cells.

    Reference: Woo et al. (2018) CBAM: Convolutional Block Attention Module.
               ECCV 2018.
    """

    def __init__(self,
                 hidden_channels: list = None,
                 kernel_size:     int  = None):
        super().__init__()
        hidden_channels = hidden_channels or config.ATTENTION_HIDDEN
        kernel_size     = kernel_size     or config.CONVLSTM_KERNEL

        # ── Encoder (identical to MeteoriteConvLSTM) ──────────────────────────
        self.encoder1 = ConvLSTMLayer(
            in_channels=1, hidden_channels=hidden_channels[0],
            kernel_size=kernel_size, return_all_seq=True)
        self.encoder2 = ConvLSTMLayer(
            in_channels=hidden_channels[0], hidden_channels=hidden_channels[1],
            kernel_size=kernel_size, return_all_seq=False)
        self.bn = nn.BatchNorm2d(hidden_channels[1])

        # ── CBAM attention gate ───────────────────────────────────────────────
        self.cbam = CBAM(channels=hidden_channels[1], reduction=8, spatial_k=7)

        # ── Decoder (identical to MeteoriteConvLSTM) ──────────────────────────
        decoder_ch = config.DECODER_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels[1], decoder_ch[0],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(decoder_ch[0]),
            nn.Conv2d(decoder_ch[0], decoder_ch[1],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch[1], decoder_ch[2], kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)     # (B, T, hidden[0], H, W)
        enc2 = self.encoder2(enc1)  # (B, hidden[1], H, W)
        enc2 = self.bn(enc2)
        enc2 = self.cbam(enc2)      # ← attention gate
        return self.decoder(enc2)   # (B, 1, H, W)


# ── PredRNN ────────────────────────────────────────────────────────────────────

class PredRNNCell(nn.Module):
    """
    PredRNN cell with spatiotemporal memory M_t (Wang et al., NeurIPS 2017).

    Standard ConvLSTM only passes H and C inside a layer.
    PredRNN additionally maintains a spatiotemporal memory M that flows
    in a zigzag pattern *across layers and timesteps*, enabling inter-layer
    memory transfer.

    Cell equations:
        g  = tanh (Wx_g * x + Wh_g * H_prev + b_g)
        i  = σ   (Wx_i * x + Wh_i * H_prev + b_i)
        f  = σ   (Wx_f * x + Wh_f * H_prev + b_f)
        C  = f ⊙ C_prev + i ⊙ g

        g' = tanh (Wx_g' * x + Wm_g' * M_prev + b_g')
        i' = σ   (Wx_i' * x + Wm_i' * M_prev + b_i')
        f' = σ   (Wx_f' * x + Wm_f' * M_prev + b_f')
        M  = f' ⊙ M_prev + i' ⊙ g'

        o  = σ   (Wx_o * x + Wh_o * H_prev + Wm_o * M + b_o)
        H  = o ⊙ tanh(W_1x1([C, M]))

    Args:
        in_channels     : C_in
        hidden_channels : C_hidden (for H, C, and M)
        kernel_size     : spatial kernel size
    """

    def __init__(self,
                 in_channels:     int,
                 hidden_channels: int,
                 kernel_size:     int = 3,
                 mem_channels:    int = None):
        super().__init__()
        self.hidden_channels = hidden_channels
        mem_channels = mem_channels if mem_channels is not None else hidden_channels
        self.mem_channels = mem_channels
        pad = kernel_size // 2

        # Standard ConvLSTM gates (input x and hidden H)
        self.conv_xh = nn.Conv2d(in_channels + hidden_channels,
                                  4 * hidden_channels,
                                  kernel_size, padding=pad, bias=True)

        # Spatiotemporal memory gates (input x and memory M)
        # mem_channels may differ from hidden_channels when M flows across
        # layers with different widths (zigzag pattern in multi-size PredRNN)
        self.conv_xm = nn.Conv2d(in_channels + mem_channels,
                                  3 * hidden_channels,
                                  kernel_size, padding=pad, bias=True)

        # Project incoming M to hidden_channels before gated update
        # (identity when mem_channels == hidden_channels)
        self.conv_mp = (nn.Conv2d(mem_channels, hidden_channels,
                                   kernel_size=1, bias=False)
                        if mem_channels != hidden_channels else None)

        # Output gate extra term: M contribution (m_t has hidden_channels)
        self.conv_mo = nn.Conv2d(hidden_channels, hidden_channels,
                                  kernel_size, padding=pad, bias=False)

        # 1×1 conv to combine [C, M] → H input
        self.conv_cm = nn.Conv2d(2 * hidden_channels, hidden_channels,
                                  kernel_size=1, bias=False)

    def forward(self,
                x_t:    torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor,
                m_prev: torch.Tensor):
        """
        Args:
            x_t, h_prev, c_prev : standard ConvLSTM tensors (B, *, H, W)
            m_prev               : spatiotemporal memory   (B, hidden, H, W)
        Returns:
            h_t, c_t, m_t
        """
        # Standard gates from x and H
        xh    = torch.cat([x_t, h_prev], dim=1)
        gates = self.conv_xh(xh)                       # 4*hidden
        g, i, f, o_partial = torch.chunk(gates, 4, dim=1)
        g = torch.tanh(g);   i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_t = f * c_prev + i * g

        # Spatiotemporal memory gates from x and M
        xm     = torch.cat([x_t, m_prev], dim=1)
        mgates = self.conv_xm(xm)                      # 3*hidden
        g2, i2, f2 = torch.chunk(mgates, 3, dim=1)
        g2 = torch.tanh(g2);  i2 = torch.sigmoid(i2)
        f2 = torch.sigmoid(f2)
        # Project m_prev to hidden_channels when sizes differ
        m_prev_h = self.conv_mp(m_prev) if self.conv_mp is not None else m_prev
        m_t = f2 * m_prev_h + i2 * g2

        # Output gate: combine partial from xh with M contribution
        o_t = torch.sigmoid(o_partial + self.conv_mo(m_t))

        # Hidden state: gated combination of C and M
        h_t = o_t * torch.tanh(self.conv_cm(torch.cat([c_t, m_t], dim=1)))
        return h_t, c_t, m_t

    def init_states(self, batch_size: int, height: int, width: int,
                    device: torch.device):
        zeros = lambda: torch.zeros(
            batch_size, self.hidden_channels, height, width, device=device)
        return zeros(), zeros(), zeros()   # h, c, m


class PredRNN(nn.Module):
    """
    Two-layer PredRNN for spatiotemporal meteorite density prediction.

    Spatiotemporal memory M flows in a zigzag pattern across layers:
        (Layer0, t=0) → (Layer1, t=0) → (Layer0, t=1) → (Layer1, t=1) → …

    This allows M to capture cross-layer temporal dependencies that
    standard ConvLSTM misses.

    Reference: Wang et al. (2017) PredRNN: Recurrent Neural Networks for
               Predictive Learning Using Spatiotemporal LSTMs. NeurIPS 2017.
    """

    def __init__(self,
                 hidden_channels: list = None,
                 kernel_size:     int  = None):
        super().__init__()
        hidden_channels = hidden_channels or config.PREDRNN_HIDDEN
        kernel_size     = kernel_size     or config.CONVLSTM_KERNEL

        self.num_layers = len(hidden_channels)
        self.hidden_ch  = hidden_channels

        # Build one PredRNNCell per layer.
        # In the zigzag M flow, layer 0 receives M from the *last* layer of
        # the previous timestep, so its mem_channels = hidden_channels[-1].
        # Every other layer receives M from the layer directly below it at
        # the same timestep, so mem_channels = hidden_channels[l-1].
        cells = []
        for l in range(self.num_layers):
            in_ch  = 1 if l == 0 else hidden_channels[l - 1]
            hid_ch = hidden_channels[l]
            mem_ch = hidden_channels[-1] if l == 0 else hidden_channels[l - 1]
            cells.append(PredRNNCell(in_ch, hid_ch, kernel_size, mem_ch))
        self.cells = nn.ModuleList(cells)

        self.bn = nn.BatchNorm2d(hidden_channels[-1])

        # Decoder identical to MeteoriteConvLSTM
        decoder_ch = config.DECODER_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], decoder_ch[0],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(decoder_ch[0]),
            nn.Conv2d(decoder_ch[0], decoder_ch[1],
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch[1], decoder_ch[2], kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 1, H, W)
        Returns:
            pred : (B, 1, H, W)
        """
        B, T, _, H, W = x.shape
        device = x.device

        # Initialise H, C, M for all layers
        states = [self.cells[l].init_states(B, H, W, device)
                  for l in range(self.num_layers)]
        # h[l], c[l], m[l] = states[l]

        # Zigzag M: flows from last layer of t-1 to first layer of t.
        # Layer 0 expects M with hidden_ch[-1] channels (the size produced
        # by the last layer), so initialise accordingly.
        m_zigzag = torch.zeros(B, self.hidden_ch[-1], H, W, device=device)

        for t in range(T):
            x_t = x[:, t]           # (B, 1, H, W)
            for l in range(self.num_layers):
                h_prev, c_prev, m_prev = states[l]

                # Layer 0 gets zigzag M from previous timestep's last layer
                # Other layers get M from current timestep's previous layer
                if l == 0:
                    m_in = m_zigzag
                else:
                    _, _, m_in = states[l - 1]   # updated in-place above

                inp = x_t if l == 0 else states[l - 1][0]   # use h of prev layer
                h_t, c_t, m_t = self.cells[l](inp, h_prev, c_prev, m_in)
                states[l] = (h_t, c_t, m_t)

            # After all layers at this timestep, last layer's M becomes
            # the zigzag memory for layer 0 at the next timestep
            m_zigzag = states[-1][2]

        # Use final hidden state of last layer
        h_final = states[-1][0]   # (B, hidden[-1], H, W)
        h_final = self.bn(h_final)
        return self.decoder(h_final)


# ── Baseline Models ────────────────────────────────────────────────────────────

class LastFrameBaseline(nn.Module):
    """Predict D_{t+1} = last cumulative frame D_t (simplest baseline)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        return x[:, -1, :, :, :]   # (B, 1, H, W) — last frame


class SingleFrameCNN(nn.Module):
    """
    Single-frame CNN baseline: uses only the most recent cumulative map.
    No temporal information.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :, :, :]    # (B, 1, H, W) — ignore all but last frame
        return self.net(last)


# ── LSTM-Flat Baseline ─────────────────────────────────────────────────────────

class LSTMFlatBaseline(nn.Module):
    """
    Flatten-LSTM baseline: flattens each spatial frame, passes through a
    standard LSTM, then reshapes + Conv output back to (B, 1, H, W).

    Treats spatial pixels as independent features — no spatial inductive bias.
    Included to demonstrate that preserving spatial structure (ConvLSTM) matters.

    Args:
        grid_h : map height  (config.GRID_H)
        grid_w : map width   (config.GRID_W)
        hidden : LSTM hidden size (capped at 512 for memory reasons)
    """

    def __init__(self,
                 grid_h:  int = None,
                 grid_w:  int = None,
                 hidden:  int = 512):
        super().__init__()
        grid_h = grid_h or config.GRID_H
        grid_w = grid_w or config.GRID_W
        self.grid_h = grid_h
        self.grid_w = grid_w
        feat_dim = grid_h * grid_w          # e.g. 90*180 = 16 200

        self.lstm = nn.LSTM(
            input_size  = feat_dim,
            hidden_size = hidden,
            num_layers  = 1,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 1, H, W)
        Returns:
            pred : (B, 1, H, W)
        """
        B, T, C, H, W = x.shape
        # Flatten spatial dims → (B, T, H*W)
        x_flat = x.view(B, T, H * W)
        out, _ = self.lstm(x_flat)          # (B, T, hidden)
        last    = out[:, -1, :]             # (B, hidden)
        pred_flat = self.head(last)         # (B, H*W)
        pred = pred_flat.view(B, 1, H, W)   # (B, 1, H, W)
        return pred


# ── Model summary helper ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, T, C, H, W = 2, config.SEQ_LEN, 1, config.GRID_H, config.GRID_W
    dummy = torch.randn(B, T, C, H, W).to(device)

    models_to_test = {
        "MeteoriteConvLSTM":  MeteoriteConvLSTM(),
        "AttentionConvLSTM":  AttentionConvLSTM(),
        "PredRNN":            PredRNN(),
        "LastFrameBaseline":  LastFrameBaseline(),
        "SingleFrameCNN":     SingleFrameCNN(),
        "LSTMFlatBaseline":   LSTMFlatBaseline(),
    }

    for name, m in models_to_test.items():
        m = m.to(device)
        out = m(dummy)
        assert out.shape == (B, 1, H, W), f"{name}: shape {out.shape} != {(B,1,H,W)}"
        n_params = count_parameters(m)
        print(f"  {name:<22s}  params={n_params:>8,}  out={tuple(out.shape)}  OK")

    print("\nAll model forward passes: OK")
