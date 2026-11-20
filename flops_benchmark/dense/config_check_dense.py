"""
Configuration check for the dense models (input size, RoI count, neck / decoder channels, backbone size).

    python config_check_dense.py det     # -> results/config_check_det.json
    python config_check_dense.py seg     # -> results/config_check_seg.json

For every model the settings are read from the built model and its config. For the UperNet models the
analysis of measure_dense.py is also run once more with `model.decode_head.channels=512` (the value of
mmsegmentation's own UperNet base configs, e.g. configs/_base_/models/upernet_swin.py); this extra number
is only part of the check and is not used in the result tables.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_dense as D  # noqa: E402

UPERNET_CHECK = {'model.decode_head.channels': 512}


def backbone_facts(bb):
    f = dict(type=type(bb).__name__, params=D.nparams(bb))
    if hasattr(bb, 'stages') and hasattr(bb, 'downsample_layers'):
        f['depths'] = [len(s) for s in bb.stages]
        f['dims'] = [d.conv.out_channels for d in bb.downsample_layers]
        f['token_mixers'] = [type(s[0].token_mixer).__name__ for s in bb.stages]
    elif hasattr(bb, 'network'):                       # SPANet (v1) detection backbone
        idx = bb.out_indices
        f['depths'] = [len(bb.network[i]) for i in idx]
        f['dims'] = [getattr(bb, f'norm{i}').num_channels for i in idx]
        f['token_mixers'] = [type(bb.network[i][0].token_mixer).__name__ for i in idx]
        f['patch_dims'] = [list(bb.network[i][0].token_mixer.sps[0].filter.shape[-2:]) for i in idx]
    for k in ('head', 'norm'):
        m = getattr(bb, k, None)
        if m is not None:
            f[f'unused_{k}_params'] = D.nparams(m)
    return f


def det_facts(model, cfg):
    tc = cfg.model.test_cfg
    rh = model.roi_head
    bbox_heads = rh.bbox_head if hasattr(rh, 'num_stages') else [rh.bbox_head]
    mask_heads = rh.mask_head if hasattr(rh, 'num_stages') else [rh.mask_head]
    return dict(
        neck=dict(type=type(model.neck).__name__, in_channels=list(model.neck.in_channels),
                  out_channels=model.neck.out_channels, num_outs=model.neck.num_outs),
        rpn_head=dict(type=type(model.rpn_head).__name__, feat_channels=model.rpn_head.feat_channels),
        roi_head=dict(type=type(rh).__name__, num_stages=getattr(rh, 'num_stages', 1),
                      bbox_heads=[f'{type(h).__name__}(convs={getattr(h, "num_shared_convs", 0)}, '
                                  f'fcs={getattr(h, "num_shared_fcs", 0)}, fc_out={getattr(h, "fc_out_channels", "-")})'
                                  for h in bbox_heads],
                      mask_heads=[f'{type(h).__name__}(convs={h.num_convs}, ch={h.conv_out_channels})' for h in mask_heads]),
        test_cfg_rpn=dict(nms_pre=tc.rpn.nms_pre, max_per_img=tc.rpn.max_per_img),
        test_cfg_rcnn_max_per_img=tc.rcnn.max_per_img,
        rois_counted=dict(bbox=D.NUM_ROIS, mask=D.NUM_MASK_ROIS))


def seg_facts(model, cfg):
    dh = model.decode_head
    f = dict(decode_head=dict(type=type(dh).__name__, in_channels=list(dh.in_channels), channels=dh.channels,
                              pool_scales=cfg.model.decode_head.get('pool_scales')))
    if getattr(model, 'neck', None) is not None:
        f['neck'] = dict(type=type(model.neck).__name__, in_channels=list(model.neck.in_channels),
                         out_channels=model.neck.out_channels, num_outs=model.neck.num_outs)
    aux = cfg.model.get('auxiliary_head')
    f['auxiliary_head_in_config'] = None if aux is None else f"{aux['type']}(channels={aux.get('channels')})"
    return f


def main():
    task = sys.argv[1]
    reg = D.setup_task(task)
    out = {}
    for entry in D.MODELS:
        setting, name, *_rest, source, ref = entry
        if D.SETTINGS[setting]['task'] != task:
            continue
        model, cfg, overrides, _ = D.build_dense(reg, ref)
        r = dict(setting=setting, name=name, input=[1, 3, *D.SETTINGS[setting]['hw']],
                 config=os.path.relpath(ref, HERE), backbone=backbone_facts(model.backbone))
        r.update(det_facts(model, cfg) if task == 'det' else seg_facts(model, cfg))
        if setting == 'upernet':
            m = D.measure_one(entry, reg, cfg_options=UPERNET_CHECK)
            r['channels512'] = dict(overrides=m['overrides'], params=m['params'], ga=m['ga'], total=m['total'],
                                    params_with_auxiliary_head=m['params_with_auxiliary_head'],
                                    a=m['a'], b=m['b'], c=m['c'], d=m['d'],
                                    ga_by_module_top=m['ga_by_module_top'])
            print(f"{name:20s} channels=512: params {m['params'] / 1e6:.3f}M ga {m['ga'] / 1e9:.3f} "
                  f"total {m['total'] / 1e9:.3f}", flush=True)
        out[f'{setting}/{name}'] = r
        print(setting, name, json.dumps(r['backbone']), flush=True)
    with open(os.path.join(D.OUT, f'config_check_{task}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == '__main__':
    main()
