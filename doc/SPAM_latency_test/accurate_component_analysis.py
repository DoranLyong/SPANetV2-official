"""
Token Mixer Component Analysis
=============================
Based on the accurate results from token_mixer_latency_analysis.py, this script
analyzes the latency breakdown of each token mixer's components.

Accurate Results (Stage 3: 320 dim, 14x14):
- SepConv:       1.430 ms (1.00x)
- ExSPAM:        2.077 ms (1.45x)  
- DynamicFilter: 3.516 ms (2.46x)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


from timm.layers import to_2tuple


# Create results directory
os.makedirs('results_figs', exist_ok=True)


def warmup_gpu(model, x, num_warmup=10):
    """GPU warmup"""
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize()


def measure_latency(model, x, num_iterations=200):
    """Measure latency with CUDA events"""
    model.eval()
    warmup_gpu(model, x)
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(x)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    
    return np.mean(latencies), np.std(latencies)

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class SepConvComponents:
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim=320):
        self.dim = dim
        med_channels = int(2 * dim)  # expansion_ratio=2
        
        # Components
        self.pwconv1 = nn.Linear(dim, med_channels, bias=False)
        self.dwconv = nn.Conv2d(med_channels, med_channels, kernel_size=7, 
                               padding=3, groups=med_channels, bias=False)
        self.pwconv2 = nn.Linear(med_channels, dim, bias=False)
        self.act1 = StarReLU()

    def to(self, device):
        self.pwconv1 = self.pwconv1.to(device)
        self.dwconv = self.dwconv.to(device)
        self.pwconv2 = self.pwconv2.to(device)
        return self

    def measure_components(self, x, device):
        """ Measure latency of each component
            x : (B, H, W, C)
        """
        results = {}
        
        # 1. First pointwise conv
        model = self.pwconv1.to(device)
        mean_lat, std_lat = measure_latency(model, x)
        results["PW Conv 1"] = {"mean": mean_lat, "std": std_lat}

        #-- 2. StarReLU activation
        x_act = self.pwconv1(x)        
        act1 = self.act1.to(device)
        mean_lat, std_lat = measure_latency(act1, x_act)
        results["act_StarReLU"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 3. Depthwise conv (need to transpose for conv2d)
        x_expanded = self.pwconv1(x)
        x_act = self.act1(x_expanded)
        x_transposed = x_act.permute(0, 3, 1, 2)  # (B, C, H, W)
        model = self.dwconv.to(device)
        mean_lat, std_lat = measure_latency(model, x_transposed)
        results["DW Conv 7x7"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 4. Second pointwise conv
        x_back = self.dwconv(x_transposed).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        model = self.pwconv2.to(device)
        mean_lat, std_lat = measure_latency(model, x_back)
        results["PW Conv 2"] = {"mean": mean_lat, "std": std_lat}
        
        return results




class SPAMComponents:
    """Break down ExSPAM into components"""
    def __init__(self, dim=320, size=14):        
        size = to_2tuple(size)
        self.size = size[0] # H
        self.filter_size = size[1] // 2 + 1 # W//2 + 1

        self.dim = dim
        self.num_heads = 4
        
        # Key components
        self.query = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU())
        self.ctx = nn.Linear(dim, dim, bias=False)
        
        # Multi-head convs
        self.split_groups=self.dim// self.num_heads
        self.split_indexes = [self.split_groups for i in range(self.num_heads)]

        self.local_convs = nn.ModuleList()
        for i in range(self.num_heads):            
            local_conv = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.num_heads)                                         
            self.local_convs.append(local_conv)
        
        # Filter for FFT
        self.filter = nn.Parameter(torch.randn(dim, self.size, self.filter_size, dtype=torch.float32) * 0.02)  # (dim, h, w)

        # Projection
        expand_ratio = 2
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups, bias=False),
            nn.GroupNorm(1, dim*expand_ratio), # layer norm for (B,C,H,W)
            nn.GELU(),
            nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1, bias=False))
        
        # -- Update -- #
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(0.)
        
        

    def to(self, device):
        self.query = self.query.to(device)
        self.ctx = self.ctx.to(device)
        self.local_convs = self.local_convs.to(device)
        self.proj = self.proj.to(device)
        self.filter = self.filter.to(device)
        return self

    def measure_components(self, x, device):
        """ Measure latency of each component
            x : (B, H, W, C)
        """
        results = {}
        
        #-- 1. Query generation
        model = self.query.to(device)
        mean_lat, std_lat = measure_latency(model, x)
        results["Linear_value"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 2. Context generation  
        model = self.ctx.to(device)
        mean_lat, std_lat = measure_latency(model, x)
        results["Linear_ctx"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 3. Multi-head convolutions (combined)
        ctx_out = self.ctx(x).permute(0,-1,1,2)  # (B, C, H, W)        
        split_groups = self.split_groups
        
        class MultiHeadConv(nn.Module):
            def __init__(self, local_convs):
                super().__init__()
                self.local_convs = local_convs
                self.num_heads = len(local_convs)
                self.split_groups = split_groups
                
            def forward(self, x):
                heads = torch.split(x, self.split_groups, dim=1)
                return torch.cat([self.local_convs[i](heads[i]) 
                                for i in range(self.num_heads)], dim=1)
        
        multihead_model = MultiHeadConv(self.local_convs).to(device)
        mean_lat, std_lat = measure_latency(multihead_model, ctx_out)
        results["Multi-scale\nConv"] = {"mean": mean_lat, "std": std_lat}

        # -- Single Conv test 
#        i = 1
#        local_conv = nn.Sequential(
#            nn.Conv2d(self.dim, self.dim, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=self.dim),
#            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0, stride=1, bias=False)
#                                                                            ).to(device)
#        
#        mean_lat, std_lat = measure_latency(local_conv, ctx_out)
#        results["Single-scale\nConv"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 4. FFT operations
        conv_feat = multihead_model(ctx_out)
        
        class FFTOp(nn.Module):
            def __init__(self, filter_param):
                super().__init__()
                self.filter = filter_param
                
            def forward(self, x):
                s_i = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
                SRF = torch.sigmoid(self.filter)
                s_i.mul_(SRF)
                return torch.fft.irfft2(s_i, s=(x.shape[2], x.shape[3]), 
                                      dim=(2, 3), norm='ortho').to(x.dtype)
        
        fft_model = FFTOp(self.filter).to(device)
        mean_lat, std_lat = measure_latency(fft_model, conv_feat)
        results["SRF"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 5. Projection
        fft_out = fft_model(conv_feat)
        model = self.proj.to(device)
        mean_lat, std_lat = measure_latency(model, fft_out)
        results["Projection"] = {"mean": mean_lat, "std": std_lat}

        #-- 6. Modulation 
        value = self.query(x)
        s_ctx = self.proj(fft_out).permute(0,2,3,1)  # Back to (B, H, W, C)

        class Modulation(nn.Module):
            def __init__(self,value):
                super().__init__()
                self.value = value                
            def forward(self, ctx):
                return self.value * ctx
        modulation_model = Modulation(value).to(device)
        mean_lat, std_lat = measure_latency(modulation_model, s_ctx)
        results["Conv_Modulation"] = {"mean": mean_lat, "std": std_lat}

        #-- 7. Linear out 
        mod_out = modulation_model(s_ctx)
        model = self.proj_out.to(device)
        mean_lat, std_lat = measure_latency(model, mod_out)
        results["Linear_out"] = {"mean": mean_lat, "std": std_lat}
        
        return results

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class DynamicFilterComponents:
    """Break down DynamicFilter into components"""
    def __init__(self, dim=320, size=14):
        self.dim = dim
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = 4
        expansion_ratio = 2
        self.med_channels = int(expansion_ratio * dim)
        
        # Components
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=False)
        self.act1 = StarReLU()
        self.reweight = Mlp(dim, .25, self.num_filters* self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, self.num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=False)

    def to(self, device):
        self.pwconv1 = self.pwconv1.to(device)
        self.reweight = self.reweight.to(device)
        self.complex_weights = self.complex_weights.to(device)
        self.pwconv2 = self.pwconv2.to(device)
        return self

    def measure_components(self, x, device):
        """Measure latency of each component"""
        B, H, W, C = x.shape
        results = {}
        
        #-- 1. First pointwise conv
        model = self.pwconv1.to(device)
        mean_lat, std_lat = measure_latency(model, x)
        results["PW Conv 1"] = {"mean": mean_lat, "std": std_lat}

        #-- 2. StarReLU activation
        x_act = self.pwconv1(x)        
        act1 = self.act1.to(device)
        mean_lat, std_lat = measure_latency(act1, x_act)
        results["act_StarReLU"] = {"mean": mean_lat, "std": std_lat}
        
        #-- 3. Routing weights
        class RoutingWeights(nn.Module):
            def __init__(self,dim, reweight_expansion_ratio=.25, num_filters=self.num_filters, med_channels=self.med_channels):
                super().__init__()
                self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters* med_channels)
                
            def forward(self, x):
                x_mean = x.mean(dim=(1, 2))  # Global average
                return self.reweight(x_mean)

        model = RoutingWeights(self.dim, .25, self.num_filters, self.med_channels).to(device)        
        mean_lat, std_lat = measure_latency(model, x)
        results["Routing "] = {"mean": mean_lat, "std": std_lat}
        
        #-- 4. Complex FFT filtering
        x_expanded = self.pwconv1(x)
        routing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters, -1).softmax(dim=1)
        
        class ComplexFFT(nn.Module):
            def __init__(self, complex_weights, routing_shape, dim):
                super().__init__()
                self.complex_weights = complex_weights
                self.routing_shape = routing_shape
                self.med_channels = int(2 * dim)
                
            def forward(self, inputs):
                x, routing = inputs
                x = x.to(torch.float32)
                B, H, W, _ = x.shape
                size = H
                filter_size = W // 2 + 1

                x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
                
                complex_weights = torch.view_as_complex(self.complex_weights)
                routing = routing.to(torch.complex64)
                weight = torch.einsum('bfc,hwf->bhwc', routing, complex_weights)
                weight = weight.view(-1, size, filter_size, self.med_channels)
                x = x * weight
                return torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        
        fft_model = ComplexFFT(self.complex_weights, routing.shape, self.dim).to(device)
        mean_lat, std_lat = measure_latency(fft_model, (x_expanded, routing))
        results["Dyn_FFT"] = {"mean": mean_lat, "std": std_lat}
        
        # 4. Second pointwise conv
        fft_out = fft_model((x_expanded, routing))
        model = self.pwconv2.to(device)
        mean_lat, std_lat = measure_latency(model, fft_out)
        results["PW Conv 2"] = {"mean": mean_lat, "std": std_lat}
        
        return results


def analyze_components():
    """Analyze components of each token mixer"""
    device = torch.device('cuda:0')
    
    # Use Stage 3 configuration (main bottleneck)
    dim, size = 320, 14    
    batch_size = 128
    
    print("=" * 60)
    print("Token Mixer Component Analysis")
    print("=" * 60)
    print(f"Configuration: dim={dim}, size={size}x{size}, batch={batch_size}")
    print()
    
    # Input tensor
    x = torch.randn(batch_size, size, size, dim, device=device, dtype=torch.float32)
    
    all_results = {}
    
    # 1. ExSPAM Components
    print("2. SPAM Components:")
    print("-" * 30)
    exspam_comp = SPAMComponents(dim, size).to(device)
    exspam_results = exspam_comp.measure_components(x, device)
    all_results["SPAM"] = exspam_results
    
    total_exspam = sum(comp["mean"] for comp in exspam_results.values())
    for comp_name, result in exspam_results.items():
        percentage = (result["mean"] / total_exspam) * 100
        print(f"{comp_name:15s}: {result['mean']:6.3f} ± {result['std']:5.3f} ms ({percentage:5.1f}%)")
    print(f"{'Total':15s}: {total_exspam:6.3f} ms")
    print()


    # 2. SepConv Components
    print("2. SepConv Components:")
    print("-" * 30)
    sepconv_comp = SepConvComponents(dim).to(device)
    sepconv_results = sepconv_comp.measure_components(x, device)
    #epconv_comp = SepConv(dim).to(device)
    #epconv_results = sepconv_comp(x)

    all_results["SepConv"] = sepconv_results
    
    total_sepconv = sum(comp["mean"] for comp in sepconv_results.values())
    for comp_name, result in sepconv_results.items():
        percentage = (result["mean"] / total_sepconv) * 100
        print(f"{comp_name:15s}: {result['mean']:6.3f} ± {result['std']:5.3f} ms ({percentage:5.1f}%)")
    print(f"{'Total':15s}: {total_sepconv:6.3f} ms")
    print()
    

    
    # 3. DynamicFilter Components  
    print("3. DynamicFilter Components:")
    print("-" * 30)
    dynfilter_comp = DynamicFilterComponents(dim, size).to(device)
    dynfilter_results = dynfilter_comp.measure_components(x, device)
    all_results["DynamicFilter"] = dynfilter_results
    
    total_dynfilter = sum(comp["mean"] for comp in dynfilter_results.values())
    for comp_name, result in dynfilter_results.items():
        percentage = (result["mean"] / total_dynfilter) * 100
        print(f"{comp_name:15s}: {result['mean']:6.3f} ± {result['std']:5.3f} ms ({percentage:5.1f}%)")
    print(f"{'Total':15s}: {total_dynfilter:6.3f} ms")
    
    return all_results


def create_component_visualizations(results):
    """Create component breakdown visualizations"""
    
    # Prepare data
    token_mixers = [ "SepConv", "DynamicFilter","SPAM"]
    colors = ['#2E8B57', '#FF6B35', '#8B0000']
    
    # 1. Component breakdown pie charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, mixer in enumerate(token_mixers):
        ax = axes[i]
        components = list(results[mixer].keys())
        latencies = [results[mixer][comp]["mean"] for comp in components]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(latencies, labels=components, autopct='%1.1f%%',
                                         startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(components))),
                                         textprops={'fontsize': 16})  # Label 폰트 크기
        
        # 퍼센트 텍스트(수치값) 폰트 크기 별도 설정
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)  # 수치값 폰트 크기 
        
        ax.set_title(f'{mixer} Latency\nTotal: {sum(latencies):.2f} ms', 
                    fontweight='bold', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('results_figs/latency_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: latency_breakdown.png")
    
    # 2. Component comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    all_components = set()
    for mixer_results in results.values():
        all_components.update(mixer_results.keys())
    all_components = sorted(list(all_components))
    
    x = np.arange(len(all_components))
    width = 0.25
    
    for i, mixer in enumerate(token_mixers):
        mixer_latencies = []
        for comp in all_components:
            if comp in results[mixer]:
                mixer_latencies.append(results[mixer][comp]["mean"])
            else:
                mixer_latencies.append(0)  # Component not present
        
        ax.bar(x + i*width, mixer_latencies, width, label=mixer, color=colors[i])
    
    ax.set_xlabel('Components')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Token Mixer Component Latency Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_components, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_figs/component_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: component_comparison_bars.png")
    
    # 3. Stacked bar chart showing total latency breakdown
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bottom_vals = [0, 0, 0]
    component_colors = plt.cm.Set3(np.linspace(0, 1, 8))  # More colors
    
    for j, mixer in enumerate(token_mixers):
        components = list(results[mixer].keys())
        latencies = [results[mixer][comp]["mean"] for comp in components]
        
        for k, (comp, lat) in enumerate(zip(components, latencies)):
            ax.bar(mixer, lat, bottom=bottom_vals[j], 
                  color=component_colors[k % len(component_colors)], 
                  label=comp if j == 0 else "")  # Only show legend once
            bottom_vals[j] += lat
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Token Mixer Total Latency Breakdown')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add total values on top
    for i, mixer in enumerate(token_mixers):
        total = bottom_vals[i]
        ax.annotate(f'{total:.2f} ms', xy=(i, total), xytext=(0, 5),
                   textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_figs/component_stacked_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: component_stacked_breakdown.png")


def main():
    """Main function"""
    if not torch.cuda.is_available():
        print("CUDA not available. This analysis requires GPU.")
        return
    
    print("Starting Token Mixer Component Analysis...")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    try:
        # Analyze components
        results = analyze_components()
        
        # Create visualizations
        create_component_visualizations(results)
        
        print("\nComponent analysis complete!")
        print("Graphs saved to results_figs/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
