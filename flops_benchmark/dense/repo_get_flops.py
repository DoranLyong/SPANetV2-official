"""
Step 2: run each repository's own FLOPs script unchanged, and log what it counted.

    # SPANetV2 detection, mmdet 3.x (env spanetv2-dense). The script reads images through the val dataloader;
    # the test pipeline's Resize is set to a fixed 1280x800 (H x W) with --cfg-options.
    python repo_get_flops.py det <config> --coco <COCO dir> --num-images 5
    # SPANetV2 segmentation, mmseg 1.x (env spanetv2-dense)
    python repo_get_flops.py seg <config> --shape 512 2048
    # DFFormer segmentation, mmseg 0.x / mmcv 1.x (env dfformer-mm0)
    python repo_get_flops.py dfformer <config> --shape 512 512

Nothing in the scripts or models is changed. This launcher only
  - fixes the random seeds,
  - registers the backbones of configs/ (ConvFormer / CAFormer / SPANet wrappers) for the SPANetV2 scripts,
  - wraps the counting function the script imports (get_model_complexity_info) to keep its raw return value,
  - adds a global forward hook that records how many RoIs reach each RoI feature extractor and the
    input shape of the backbone (two-stage detectors).
The last stdout line is `RESULT_JSON {...}`.
"""
import json
import os
import random
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.dirname(HERE)
ROOT = os.path.dirname(FB)
TP = os.path.join(FB, 'third_party')

SCRIPTS = {
    'det': (os.path.join(ROOT, 'object_detection'), 'tools/analysis_tools/get_flops.py'),
    'seg': (os.path.join(ROOT, 'semantic_segmentation'), 'tools/analysis_tools/get_flops.py'),
    'dfformer': (os.path.join(TP, 'dfformer'), 'segmentation/tools/get_flops.py'),
}

RAW, EVENTS = [], []


def main():
    task, config, rest = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3:]
    coco = None
    if '--coco' in rest:
        i = rest.index('--coco')
        coco = os.path.abspath(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    repo, script = SCRIPTS[task]
    os.chdir(repo)                        # the repositories' shell scripts run from here with PYTHONPATH=$(pwd)
    sys.path.insert(0, repo)
    sys.path.insert(1, HERE)

    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    argv = [script, config]
    if task in ('det', 'seg'):
        import backbones
        if task == 'det':
            import mmdet.models  # noqa: F401
            from mmdet.registry import MODELS as REG
        else:
            import mmseg.models  # noqa: F401
            from mmseg.registry import MODELS as REG
            with backbones.captured_stdout():
                backbones.register_spanet_v1()
        backbones.register_metaformer(REG)

        import mmengine.analysis as MA
        orig = MA.get_model_complexity_info

        def wrapped(*a, **k):
            EVENTS.append(['call'])
            out = orig(*a, **k)
            RAW.append(dict(flops=int(out['flops']), params=int(out['params']),
                            activations=int(out['activations'])))
            return out
        MA.get_model_complexity_info = wrapped

        opts = ['model.backbone.init_cfg=None']
        if task == 'det':
            assert coco, '--coco <dir with annotations/instances_val2017.json and val2017/>'
            opts += [f'val_dataloader.dataset.data_root={coco}/',
                     'val_dataloader.num_workers=0', 'val_dataloader.persistent_workers=False',
                     'val_dataloader.dataset.pipeline.1.scale=(800,1280)',      # mmdet Resize scale = (w, h)
                     'val_dataloader.dataset.pipeline.1.keep_ratio=False']
        argv += rest + ['--cfg-options'] + opts
    else:
        # torch 1.12.1+cu116's cuFFT fails on this GPU (RTX 4090, sm_89: "cuFFT error: CUFFT_INTERNAL_ERROR" at
        # torch.fft.rfft2). The model stays on the GPU as the script wants (SyncBN only runs on CUDA); only the
        # two FFT calls are computed on the CPU and moved back. mmcv 1.x's counter is hook based on module
        # types, and torch.fft.* are functions, not modules, so this does not change what it counts.
        import torch.fft as TF
        for fn in ('rfft2', 'irfft2'):
            def cpu_fft(x, *a, _orig=getattr(TF, fn), **k):
                return _orig(x.cpu(), *a, **k).to(x.device)
            setattr(TF, fn, cpu_fft)
        EVENTS.append(['note', 'torch.fft.rfft2 / irfft2 computed on the CPU (cuFFT of CUDA 11.6 fails on sm_89)'])
        import mmcv.cnn as MC
        orig = MC.get_model_complexity_info

        def wrapped(model, input_shape, *a, **k):
            EVENTS.append(['call'])
            out = orig(model, input_shape, *a, **k)           # the script's own call (strings, per-layer table)
            raw = orig(model, input_shape, print_per_layer_stat=False, as_strings=False)
            RAW.append(dict(flops=int(raw[0]), params=int(raw[1]), strings=[str(o) for o in out]))
            return out
        MC.get_model_complexity_info = wrapped
        argv += rest

    def hook(module, inputs, output):
        name = type(module).__name__
        if name.endswith('RoIExtractor') and len(inputs) >= 2 and hasattr(inputs[1], 'shape'):
            out_size = getattr(module.roi_layers[0], 'output_size', None)   # (7, 7): bbox head, (14, 14): mask head
            EVENTS.append(['roi_extractor', name, int(inputs[1].shape[0]),
                           [int(s) for s in out_size] if out_size is not None else None])
        elif name == 'RPNHead':
            EVENTS.append(['rpn_head_input', [[int(s) for s in t.shape] for t in inputs[0]]])
    if task == 'det':
        torch.nn.modules.module.register_module_forward_hook(hook)

    sys.argv = argv
    print('ARGV', json.dumps(argv), flush=True)
    runpy.run_path(script, run_name='__main__')
    print('RESULT_JSON', json.dumps(dict(task=task, config=config, argv=argv, raw=RAW, events=EVENTS)), flush=True)


if __name__ == '__main__':
    main()
