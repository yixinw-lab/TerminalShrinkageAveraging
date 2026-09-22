#!/usr/bin/env python3
"""Average lightweight NanoChat snapshots and emit a standard checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import Any
import torch


def selector(name: str, which: str, n_layers: int | None = None) -> bool:
    low=name.lower(); which=which.lower()
    m=re.search(r'(?:^|\.)h\.(\d+)(?:\.|$)',name); layer=int(m.group(1)) if m else None
    if which in {'all','*'}: return True
    if which=='muon': return name.startswith('transformer.h.')
    if which=='adamw': return not name.startswith('transformer.h.')
    if which=='attention': return '.attn.' in low or '.attention.' in low
    if which=='mlp': return '.mlp.' in low or '.ffn.' in low
    if which=='embedding': return 'wte' in low or 'value_embed' in low
    if which=='lm_head': return 'lm_head' in low
    if which in {'early','middle','late'} and layer is not None and n_layers:
        f=(layer+.5)/n_layers
        return f<=1/3 if which=='early' else (1/3<f<=2/3 if which=='middle' else f>2/3)
    return False


def choose(paths: list[Path], window: int, spacing: int) -> list[Path]:
    if len(paths)<1+(window-1)*spacing: raise SystemExit('not enough snapshots for requested window/spacing')
    return [paths[-1-j*spacing] for j in reversed(range(window))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot-dir',type=Path,required=True);ap.add_argument('--window',type=int,default=8);ap.add_argument('--spacing',type=int,default=1)
    ap.add_argument('--selector',default='all');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--output-step',type=int,default=999999);ap.add_argument('--tag-note',default='LAWA')
    a=ap.parse_args();paths=sorted(a.snapshot_dir.glob('step_*.pt'));chosen=choose(paths,a.window,a.spacing);print('selected:',*[p.name for p in chosen])
    snaps=[torch.load(p,map_location='cpu',weights_only=False) for p in chosen];latest=snaps[-1];states=[s['model'] for s in snaps]
    n_layers=None;cfg=latest.get('model_config',{});n_layers=cfg.get('n_layer',cfg.get('depth'))
    out={}
    for key,vlast in states[-1].items():
        if torch.is_floating_point(vlast) and selector(key,a.selector,n_layers):
            acc=torch.zeros_like(vlast,dtype=torch.float32)
            for s in states:acc.add_(s[key].float(),alpha=1/len(states))
            out[key]=acc.to(vlast.dtype)
        else:out[key]=vlast
    a.output_dir.mkdir(parents=True,exist_ok=True)
    from nanochat.checkpoint_manager import save_checkpoint
    meta={'step':a.output_step,'model_config':latest.get('model_config',{}),'user_config':latest.get('user_config',{}),
          'lawa':{'window':a.window,'spacing_in_saved_snapshots':a.spacing,'selector':a.selector,'sources':[p.name for p in chosen],'note':a.tag_note}}
    save_checkpoint(str(a.output_dir),a.output_step,out,None,meta,rank=0)
    print('wrote standard NanoChat checkpoint',a.output_dir,'step',a.output_step)
if __name__=='__main__':main()
