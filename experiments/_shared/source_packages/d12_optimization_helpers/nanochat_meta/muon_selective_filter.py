#!/usr/bin/env python3
"""Live Muon-aware selective trajectory filters with exact continuation.

This is the genuinely optimizer-like counterpart to evaluation-only LAWA. The
filter acts only on selected parameter groups (e.g. Muon matrices or attention
blocks), can use a causal tensorwise averaging strength, and then switches off
so that durability can be measured under ordinary exact training.

Arm grammar
-----------
exact_lr:<lr_scale>
noop
pull:<selector>:<window>:<spacing>:<period>:<alpha>:<state>
apull:<rule>:<selector>:<window>:<spacing>:<period>:<strength>:<state>

State policies: preserve, reset_m, reset_all, damp_m0.25, damp_m0.5, damp_m0.75.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .skip_core import Timer, apply_lr_schedule, seed_everything
from .skip_suite import PACK_VERSION, exact_gradient, restore_branch, arm_slug
from .filter_suite import (
    BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER, DIRECT_ASSIGN_OPS_PER_PARAMETER,
    evaluate_detailed, _install_canonical_build, _stabilize_nanochat_adamw_kernel,
)
from .trajectory_groups import (
    SparseSnapshotBank, average_selected, build_slice_infos, make_averaged_candidate,
    selected_params, tensor_stats, transport_selected_state,
)


def parse_arm(arm: str) -> tuple[str,list[str]]:
    p=arm.split(":"); return p[0],p[1:]


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16":torch.bfloat16,"float16":torch.float16,"float32":torch.float32}[name]


def _micro_tokens(args: argparse.Namespace) -> int:
    return int(args.device_batch_size)*int(args.max_seq_len)


def _lr_scale(arm: str, runtime: argparse.Namespace) -> float:
    name,x=parse_arm(arm)
    return float(x[0]) if name=="exact_lr" else float(runtime.base_lr_scale)


def _apply_lr(opt: torch.optim.Optimizer, step: int, args: argparse.Namespace, runtime: argparse.Namespace,
              scale: float) -> None:
    wr=args.warmdown_ratio if runtime.warmdown_ratio is None else runtime.warmdown_ratio
    ff=args.final_lr_frac if runtime.final_lr_frac is None else runtime.final_lr_frac
    apply_lr_schedule(opt,step,args.warmup_steps,args.total_iterations,wr,ff)
    for g in opt.param_groups: g["lr"]*=float(scale)


def arm_requirements(arm: str) -> tuple[int,int,int]:
    name,x=parse_arm(arm)
    if name in {"exact_lr","noop"}: return 1,1,10**9
    if name=="pull": return int(x[1]),int(x[2]),int(x[3])
    if name=="apull": return int(x[2]),int(x[3]),int(x[4])
    raise ValueError(arm)


def run(runtime: argparse.Namespace) -> Path:
    _stabilize_nanochat_adamw_kernel()
    pack=torch.load(runtime.pack,map_location="cpu",weights_only=False)
    if int(pack.get("pack_version",-1))!=PACK_VERSION: raise RuntimeError("pack version mismatch")
    args,model,opt,token_bytes,device,spec,_hist,_bank=restore_branch(pack,runtime)
    seed_everything(int(pack.get("seed",1337)) if runtime.seed is None else int(runtime.seed))
    infos=build_slice_infos(spec,opt)
    n_layers=int(getattr(model.config,"n_layer",getattr(model.config,"depth",args.depth)))
    name,x=parse_arm(runtime.arm)
    window,spacing,period=arm_requirements(runtime.arm)
    max_span=1+(window-1)*spacing
    max_snaps=max(8,math.ceil(max_span/runtime.snapshot_every)+16)
    exact_bank=SparseSnapshotBank(max_snaps,runtime.snapshot_every,_dtype(runtime.snapshot_dtype))
    all_bank=SparseSnapshotBank(max_snaps,runtime.snapshot_every,_dtype(runtime.snapshot_dtype))
    theta=spec.flatten_parameters(dtype=torch.float32)
    exact_bank.append(0,theta,force=True); all_bank.append(0,theta,force=True)

    micro=_micro_tokens(args); full=micro*int(args.grad_accum)
    ledger=BudgetLedger(spec.numel,full)
    stream=pack["branch_batches"]; cursor=0
    trace=[]; events=[]; next_eval=0.0; exact_since=0
    filter_stop_row=None

    def record() -> dict[str,Any]:
        nonlocal next_eval
        with Timer() as et: ev=evaluate_detailed(model,pack["eval_batches"],token_bytes,device)
        ledger.eval_seconds+=et.seconds
        row={"step":ledger.optimizer_steps,"filter_active":runtime.filter_start_equiv<=ledger.charged_flop_equiv<runtime.filter_stop_equiv,
             **ev,**ledger.as_dict()}
        trace.append(row); next_eval+=runtime.eval_every_equiv
        print(f"[{runtime.arm}] F={ledger.charged_flop_equiv:8.2f} step={ledger.optimizer_steps:5d} actions={ledger.synthetic_steps:4d} bpb={ev['bpb']:.6f}")
        return row

    record(); termination="budget"
    while ledger.charged_flop_equiv<runtime.compute_budget:
        k=int(args.grad_accum)
        if cursor+k>len(stream): termination="stream_exhausted"; break
        batches=stream[cursor:cursor+k]; cursor+=k
        global_step=int(pack["prefix_steps"])+ledger.optimizer_steps
        _apply_lr(opt,global_step,args,runtime,_lr_scale(runtime.arm, runtime))
        before=spec.flatten_parameters(dtype=torch.float32)
        with Timer() as tt:
            _g,loss,_,_=exact_gradient(model,opt,spec,batches,device,need_forward_features=False,token_sample_max=args.token_sample_max)
            opt.step(); opt.zero_grad(set_to_none=True)
        ledger.add_exact(full,k,tt.seconds,charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER*spec.numel,charged=True)
        ledger.optimizer_steps+=1; exact_since+=1
        after=spec.flatten_parameters(dtype=torch.float32)
        exact_bank.append(ledger.optimizer_steps,after)
        all_bank.append(ledger.optimizer_steps,after)

        active=(runtime.filter_start_equiv<=ledger.charged_flop_equiv<runtime.filter_stop_equiv)
        planned=name not in {"exact_lr","noop"} and active and exact_since>=period
        if planned:
            exact_bank.append(ledger.optimizer_steps,after,force=True)
            try:
                rows=exact_bank.select(window,spacing,end_step=ledger.optimizer_steps)
                avg=average_selected(rows,dtype=torch.float32)
                if name=="pull":
                    selector=x[0]; alpha=float(x[4]); state=x[5]
                    cand,alpha_rows=make_averaged_candidate(after,avg,infos,selector=selector,n_layers=n_layers)
                    cand=after+alpha*(cand-after)
                else:
                    rule=x[0]; selector=x[1]; strength=float(x[5]); state=x[6]
                    stats=tensor_stats(rows,infos)
                    cand,alpha_rows=make_averaged_candidate(after,avg,infos,n_layers=n_layers,
                        adaptive_rule=rule,adaptive_strength=strength,stats_rows=stats,adaptive_selector=selector)
                spec.assign_parameters(cand)
                params=selected_params(spec,infos,selector,n_layers)
                touched=transport_selected_state(opt,params,state)
                all_bank.append(ledger.optimizer_steps,cand,force=True)
                ledger.add_parameter_ops((DIRECT_ASSIGN_OPS_PER_PARAMETER*spec.numel)+touched,charged=True)
                ledger.synthetic_steps+=1; exact_since=0
                events.append({"kind":"filter","step":ledger.optimizer_steps,"selector":selector,"state":state,
                    "delta_norm":float((cand-after).norm()),"mean_alpha":float(np.average(
                        [r["alpha"] for r in alpha_rows],weights=[max(r["numel"],1) for r in alpha_rows]))})
            except Exception as exc:
                events.append({"kind":"filter_failure","step":ledger.optimizer_steps,"error":f"{type(exc).__name__}: {exc}"})
                exact_since=0

        if filter_stop_row is None and ledger.charged_flop_equiv>=runtime.filter_stop_equiv:
            filter_stop_row=record()
        elif ledger.charged_flop_equiv>=next_eval:
            record()

    if not trace or trace[-1]["charged_flop_equiv"]!=ledger.charged_flop_equiv: record()
    runtime.out.mkdir(parents=True,exist_ok=True)
    result={"arm":runtime.arm,"arm_slug":arm_slug(runtime.arm),"optimizer":pack["build_args"].get("optimizer"),
        "prefix_steps":int(pack["prefix_steps"]),"runtime_args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(runtime).items()},
        "build_args":pack["build_args"],"trace":trace,"events":events,"filter_stop_row":filter_stop_row,
        "final_ledger":ledger.as_dict(),"termination":termination}
    path=runtime.out/"result.json"; path.write_text(json.dumps(result,indent=2,allow_nan=True)); print("[result] wrote",path); return path


def selftest() -> None:
    from .trajectory_groups import selftest as st
    st();
    for a in ["pull:muon:4:16:8:0.25:preserve","apull:hybrid:muon:8:16:8:1.0:reset_m"]:
        w,s,p=arm_requirements(a); assert w>0 and s>0 and p>0
    print("[selftest] muon selective filter PASS")


def parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    sub.add_parser("selftest")
    bp=sub.add_parser("branch")
    bp.add_argument("--pack",type=Path,required=True); bp.add_argument("--arm",required=True); bp.add_argument("--out",type=Path,required=True)
    bp.add_argument("--device",default="cuda"); bp.add_argument("--history-device",default=None); bp.add_argument("--history-dtype",default=None)
    bp.add_argument("--compute-budget",type=float,default=1800); bp.add_argument("--base-lr-scale",type=float,default=0.30); bp.add_argument("--filter-start-equiv",type=float,default=0)
    bp.add_argument("--filter-stop-equiv",type=float,default=1200); bp.add_argument("--eval-every-equiv",type=float,default=100)
    bp.add_argument("--snapshot-every",type=int,default=4); bp.add_argument("--snapshot-dtype",default="bfloat16")
    bp.add_argument("--warmdown-ratio",type=float,default=None); bp.add_argument("--final-lr-frac",type=float,default=None)
    bp.add_argument("--seed",type=int,default=None)
    return ap


def main():
    a=parser().parse_args()
    if a.command=="selftest": selftest(); return
    _install_canonical_build(); run(a)
if __name__=="__main__": main()
