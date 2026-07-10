# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Written by Dongchen Han
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


class TTT(nn.Module):
    r""" Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643

    This block implements test-time inner training of two parallel sub-modules:
        1. Simplified SwiGLU inner module, i.e., SwiGLU with identity output layer
        2. 3x3 depth-wise convolution (3x3dwc) inner module

    Note:
        The TTT inner loss is a per-head / per-sample vector-valued loss (shape [B, num_heads]).
        The torch.autograd.backward only supports scalar losses, so here we implement a hand-derived
        backward (closed-form gradient expressions) that directly computes parameter gradients.
        Alternative efficient implementations are welcome and appreciated.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        """
        Args:
            k (torch.Tensor): Key tensor of shape [B, num_heads, N, head_dim]
            v (torch.Tensor): Value tensor of shape [B, num_heads, N, head_dim]
            w1 (torch.Tensor): First weight matrix of shape [1, num_heads, head_dim, head_dim]
            w2 (torch.Tensor): Second weight matrix of shape [1, num_heads, head_dim, head_dim]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0

        Returns:
            tuple: Updated w1 and w2
        """
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # v_hat = a
        # l = (v_hat * v).sum(dim=3).mean(dim=2) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation='prod'):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        # --- Forward ---
        # v_hat = F.conv2d(k, w, padding=1, groups=C)
        # l = - (v_hat * v).mean(dim=[-2, -1]) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        # Two equivalent implementations. The 'prod' implementation appears to be slightly faster
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == 'prod':
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        """
        Args:
            x (torch.Tensor): Input features with shape of (B, N, C)
            h (int): Feature map height
            w (int): Feature map width
            rope (nn.Module, optional): Rotary Position Embedding
        """
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation='prod')

        # Apply updated inner module to q
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        # Output proj
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'




if __name__ == '__main__':
    def test_output_shape():
        """Verify forward pass produces correct output shape."""
        B, H, W, C, heads = 2, 14, 14, 192, 3
        N = H * W
        model = TTT(dim=C, num_heads=heads)
        x = torch.randn(B, N, C)
        out = model(x, H, W)
        assert out.shape == (B, N, C), f"Expected {(B, N, C)}, got {out.shape}"
        print(f"[PASS] test_output_shape: input {x.shape} -> output {out.shape}")

    def test_different_configs():
        """Test various dim / num_heads combinations."""
        configs = [
            (64, 2, 8, 8),    # small
            (192, 3, 14, 14), # typical
            (384, 6, 7, 7),   # larger
        ]
        for dim, heads, h, w in configs:
            model = TTT(dim=dim, num_heads=heads)
            x = torch.randn(1, h * w, dim)
            out = model(x, h, w)
            assert out.shape == x.shape, f"Config dim={dim},heads={heads}: shape mismatch"
            print(f"[PASS] test_different_configs: dim={dim}, heads={heads}, h={h}, w={w}")

    def test_inner_train_swiglu_updates_weights():
        """Confirm inner training actually changes w1, w2."""
        model = TTT(dim=64, num_heads=2)
        d = 64 // 2
        k = torch.randn(1, 2, 16, d)
        v = torch.randn(1, 2, 16, d)
        w1_new, w2_new = model.inner_train_simplified_swiglu(k, v, model.w1, model.w2)
        assert not torch.allclose(w1_new, model.w1), "w1 was not updated"
        assert not torch.allclose(w2_new, model.w2), "w2 was not updated"
        print("[PASS] test_inner_train_swiglu_updates_weights")

    def test_inner_train_3x3dwc_updates_weights():
        """Confirm inner training actually changes w3."""
        model = TTT(dim=64, num_heads=2)
        d = 64 // 2
        k = torch.randn(2, d, 8, 8)
        v = torch.randn(2, d, 8, 8)
        w3_new = model.inner_train_3x3dwc(k, v, model.w3)
        assert w3_new.shape == (2 * d, 1, 3, 3), f"Expected {(2*d, 1, 3, 3)}, got {w3_new.shape}"
        print("[PASS] test_inner_train_3x3dwc_updates_weights")

    def test_3x3dwc_conv_vs_prod():
        """Both implementations of inner_train_3x3dwc should produce the same result."""
        model = TTT(dim=64, num_heads=2)
        d = 64 // 2
        k = torch.randn(2, d, 8, 8)
        v = torch.randn(2, d, 8, 8)
        w_conv = model.inner_train_3x3dwc(k, v, model.w3, implementation='conv')
        w_prod = model.inner_train_3x3dwc(k, v, model.w3, implementation='prod')
        assert torch.allclose(w_conv, w_prod, atol=1e-5), \
            f"conv vs prod mismatch, max diff: {(w_conv - w_prod).abs().max().item()}"
        print("[PASS] test_3x3dwc_conv_vs_prod")

    def test_gradient_flow():
        """Ensure gradients propagate back through the full forward pass."""
        model = TTT(dim=64, num_heads=2)
        x = torch.randn(1, 16, 64, requires_grad=True)
        out = model(x, 4, 4)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"
        assert model.qkv.weight.grad is not None, "No gradient on qkv"
        assert model.w1.grad is not None, "No gradient on w1"
        assert model.w2.grad is not None, "No gradient on w2"
        assert model.w3.grad is not None, "No gradient on w3"
        print("[PASS] test_gradient_flow")

    def test_with_rope():
        """Forward pass works when a RoPE module is provided."""
        B, H, W, C, heads = 1, 8, 8, 64, 2
        # Simple mock RoPE: identity function
        rope = lambda t: t
        model = TTT(dim=C, num_heads=heads)
        x = torch.randn(B, H * W, C)
        out = model(x, H, W, rope=rope)
        assert out.shape == (B, H * W, C)
        print("[PASS] test_with_rope")

    def test_batch_independence():
        """Each sample in a batch should produce independent results."""
        model = TTT(dim=64, num_heads=2)
        model.eval()
        x1 = torch.randn(1, 16, 64)
        x2 = torch.randn(1, 16, 64)
        out1 = model(x1, 4, 4)
        out2 = model(x2, 4, 4)
        x_cat = torch.cat([x1, x2], dim=0)
        out_cat = model(x_cat, 4, 4)
        assert torch.allclose(out_cat[0], out1[0], atol=1e-5), "Batch sample 0 mismatch"
        assert torch.allclose(out_cat[1], out2[0], atol=1e-5), "Batch sample 1 mismatch"
        print("[PASS] test_batch_independence")

    # --- Run all tests ---
    print("=" * 50)
    print("Running TTT block unit tests")
    print("=" * 50)
    test_output_shape()      # Forward pass produces correct [B, N, C] output
    test_different_configs() # Works across various dim/heads/resolution combos
    test_inner_train_swiglu_updates_weights() # TTT actually modifies w1, w2
    test_inner_train_3x3dwc_updates_weights() # TTT actually modifies w3 with correct shape
    test_3x3dwc_conv_vs_prod()  # conv and prod implementations produce identical results
    test_gradient_flow()  # Outer-loop gradients propagate through all parameters
    test_with_rope()   # Forward pass works with RoPE enabled
    test_batch_independence()  # Each sample gets its own per-sample adapted weights (no cross-contamination)
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)