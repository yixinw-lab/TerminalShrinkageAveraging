#!/usr/bin/env python3
"""Forward-verified and quadratic-probe causal fast-forwarding.

Open-loop history extrapolation failed in the previous screen. This runner uses a
small current-batch forward probe to accept, reject, or shrink a causal trajectory
proposal before deciding whether to skip the expensive backward pass.

Arm grammar
-----------
exact_lr:<lr_scale>
eval_lawa_lr:<lr_scale>:<window>:<spacing>
vf:<method>:<order>:<K>:<period>:<amax>:<state>:<clock>:<probe_micro>:<margin>:<selector>
qf:<method>:<order>:<K>:<period>:<amax>:<state>:<clock>:<probe_micro>:<margin>:<selector>

vf = two-point forward verification (alpha 0 and amax)
qf = directional quadratic probe (0, amax/2, amax, then optional fitted vertex)
method = mean, ar, dmd, recent_dmd
clock = hold or advance
selector = all, muon, adamw, attention, mlp, muon_attention, muon_mlp
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .skip_core import Timer, apply_lr_schedule, cosine, relative_error, seed_everything
from .skip_suite import PACK_VERSION, exact_gradient, restore_branch, arm_slug, _dtype_from_name
from .filter_suite import (
    BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER, DIRECT_ASSIGN_OPS_PER_PARAMETER,
    evaluate_detailed, _install_canonical_build, _stabilize_nanochat_adamw_kernel,
    _init_update_history, _mean_weight_delta,
)
from .causal_fastforward import FFSpec, forecast_delta, transport_optimizer_state
from .trajectory_groups import build_slice_infos, selector_matches, selected_params, transport_selected_state


@dataclass(frozen=True)
class ProbeSpec:
    mode: str
    method: str
    order: int
    horizon: int
    period: int
    amax: float
    state: str
    clock: str
    probe_micro: int
    margin: float
    selector: str


def parse_arm(arm: str) -> tuple[str,list[str]]:
    p=arm.split(":");return p[0],p[1:]


def parse_probe(arm: str) -> ProbeSpec:
    name,x=parse_arm(arm)
    if name not in {"vf","qf"} or len(x)!=10:
        raise ValueError("vf/qf:<method>:<order>:<K>:<period>:<amax>:<state>:<clock>:<probe_micro>:<margin>:<selector>")
    return ProbeSpec(name,x[0],int(x[1]),int(x[2]),int(x[3]),float(x[4]),x[5],x[6],int(x[7]),float(x[8]),x[9])


def arm_lr(arm:str, runtime: argparse.Namespace)->float:
    name,x=parse_arm(arm)
    return float(x[0]) if name in {"exact_lr","eval_lawa_lr"} else float(runtime.base_lr_scale)


def proposal_ffspec(ps:ProbeSpec)->FFSpec:
    method=ps.method
    if method=="recent_dmd":
        return FFSpec("recent_dmd",ps.order,ps.horizon,ps.period,1.0,ps.state,ps.clock,recency=.95,radius=.99)
    return FFSpec(method,ps.order,ps.horizon,ps.period,1.0,ps.state,ps.clock)


def _micro_tokens(args:argparse.Namespace)->int:
    return int(args.device_batch_size)*int(args.max_seq_len)


def _apply_lr(opt,step,args,runtime,scale):
    wr=args.warmdown_ratio if runtime.warmdown_ratio is None else runtime.warmdown_ratio
    ff=args.final_lr_frac if runtime.final_lr_frac is None else runtime.final_lr_frac
    apply_lr_schedule(opt,step,args.warmup_steps,args.total_iterations,wr,ff)
    for g in opt.param_groups:g["lr"]*=float(scale)


@torch.no_grad()
def probe_loss(model:torch.nn.Module,batches:Sequence[tuple[torch.Tensor,torch.Tensor]],device:torch.device)->float:
    was=model.training;model.eval();vals=[]
    for xc,yc in batches:
        x=xc.to(device,non_blocking=True);y=yc.to(device,non_blocking=True)
        loss=model(x,y);vals.append(float(loss.mean() if loss.ndim else loss))
    if was:model.train()
    return float(np.mean(vals))


def charge_probe(ledger:BudgetLedger,tokens:int,seconds:float)->None:
    ledger.forward_tokens_actual+=int(tokens);ledger.forward_tokens_charged+=int(tokens);ledger.model_seconds+=seconds


def mask_direction(direction:torch.Tensor,infos:Sequence[Any],selector:str,n_layers:int)->torch.Tensor:
    if selector in {"all","*"}:return direction
    out=torch.zeros_like(direction)
    for info in infos:
        if selector_matches(info,selector,n_layers):out[info.start:info.stop]=direction[info.start:info.stop]
    return out


def fitted_alpha(l0:float,lm:float,l1:float,amax:float)->float:
    # Fit q(a)=c+b*a+c2*a^2 through a=0, amax/2, amax.
    if amax<=0:return 0.0
    m=amax/2
    c=l0
    # Solve two equations robustly.
    A=np.array([[m,m*m],[amax,amax*amax]],dtype=float);y=np.array([lm-c,l1-c],dtype=float)
    try:b,c2=np.linalg.solve(A,y)
    except np.linalg.LinAlgError:return amax if l1<l0 else m if lm<l0 else 0.0
    if c2>0 and math.isfinite(b) and math.isfinite(c2):
        return float(np.clip(-b/(2*c2),0,amax))
    return amax if l1<lm else m


def run(runtime:argparse.Namespace)->Path:
    _stabilize_nanochat_adamw_kernel();pack=torch.load(runtime.pack,map_location="cpu",weights_only=False)
    if int(pack.get("pack_version",-1))!=PACK_VERSION:raise RuntimeError("pack version mismatch")
    args,model,opt,token_bytes,device,spec,_gh,_pb=restore_branch(pack,runtime)
    seed_everything(int(pack.get("seed",1337)) if runtime.seed is None else int(runtime.seed))
    infos=build_slice_infos(spec,opt);n_layers=int(getattr(model.config,"n_layer",getattr(model.config,"depth",args.depth)))
    name,x=parse_arm(runtime.arm);ps=parse_probe(runtime.arm) if name in {"vf","qf"} else None
    update_history,weight_history=_init_update_history(pack,args,spec,device)
    exact_weights=deque(weight_history,maxlen=max(runtime.lawa_window,256))
    if not exact_weights:exact_weights.append(spec.flatten_parameters(dtype=torch.float32).cpu().to(_dtype_from_name(args.history_dtype)))
    stream=pack["branch_batches"];cursor=0;virtual=0;exact_since=0;actions=0;accepted=0;rejected=0;skipped_tokens=0
    micro=_micro_tokens(args);full=micro*int(args.grad_accum);ledger=BudgetLedger(spec.numel,full)
    trace=[];events=[];next_eval=0.0

    def record():
        nonlocal next_eval
        current=spec.flatten_parameters(dtype=torch.float32)
        if name=="eval_lawa_lr":
            window,spacing=int(x[1]),int(x[2]);
            with Timer() as et:
                raw=evaluate_detailed(model,pack["eval_batches"],token_bytes,device)
                try:
                    delta=_mean_weight_delta(exact_weights,current,window,spacing);spec.assign_parameters(current+delta)
                    ev=evaluate_detailed(model,pack["eval_batches"],token_bytes,device)
                finally:spec.assign_parameters(current)
            ev["raw_bpb"]=raw["bpb"];ev["raw_bpb_values"]=raw["bpb_values"]
        else:
            with Timer() as et:ev=evaluate_detailed(model,pack["eval_batches"],token_bytes,device)
        ledger.eval_seconds+=et.seconds
        row={"virtual_steps":virtual,"accepted":accepted,"rejected":rejected,"jump_actions":actions,"skipped_tokens":skipped_tokens,**ev,**ledger.as_dict()}
        trace.append(row);next_eval+=runtime.eval_every_equiv
        print(f"[{runtime.arm}] F={ledger.charged_flop_equiv:8.2f} v={virtual:5d} bw={ledger.exact_backward_calls:5d} acc={accepted:4d}/{actions:4d} bpb={ev['bpb']:.6f}")

    record();termination="budget"
    while ledger.charged_flop_equiv<runtime.compute_budget:
        global_step=int(pack["prefix_steps"])+virtual;_apply_lr(opt,global_step,args,runtime,arm_lr(runtime.arm, runtime))
        enough=ps is not None and len(update_history)>=max(ps.order+1,3)
        active=ps is not None and runtime.jump_start_equiv<=ledger.charged_flop_equiv<runtime.jump_stop_equiv
        planned=active and enough and exact_since>=ps.period
        if planned:
            skip=ps.horizon*int(args.grad_accum)
            if ps.clock=="advance" and cursor+skip>len(stream):termination="stream_exhausted";break
            if cursor+ps.probe_micro>len(stream):termination="stream_exhausted";break
            probe=stream[cursor:cursor+ps.probe_micro]
            current=spec.flatten_parameters(dtype=torch.float32)
            try:
                with Timer() as pt:direction,meta=forecast_delta(update_history,proposal_ffspec(ps),ps.horizon)
                ledger.predictor_seconds+=pt.seconds
                direction=mask_direction(direction,infos,ps.selector,n_layers)
                if not torch.isfinite(direction).all():raise FloatingPointError("non-finite proposal")
                # Base probe.
                with Timer() as t0:l0=probe_loss(model,probe,device)
                charge_probe(ledger,ps.probe_micro*micro,t0.seconds)
                chosen=ps.amax;losses={"0":l0}
                if ps.mode=="vf":
                    spec.assign_parameters(current+ps.amax*direction)
                    with Timer() as t1:l1=probe_loss(model,probe,device)
                    charge_probe(ledger,ps.probe_micro*micro,t1.seconds);losses[str(ps.amax)]=l1
                    accept=l1<=l0-ps.margin
                else:
                    mid=ps.amax/2
                    spec.assign_parameters(current+mid*direction)
                    with Timer() as tm:lm=probe_loss(model,probe,device)
                    charge_probe(ledger,ps.probe_micro*micro,tm.seconds);losses[str(mid)]=lm
                    spec.assign_parameters(current+ps.amax*direction)
                    with Timer() as t1:l1=probe_loss(model,probe,device)
                    charge_probe(ledger,ps.probe_micro*micro,t1.seconds);losses[str(ps.amax)]=l1
                    chosen=fitted_alpha(l0,lm,l1,ps.amax)
                    if chosen not in {0.0,mid,ps.amax}:
                        spec.assign_parameters(current+chosen*direction)
                        with Timer() as tv:lv=probe_loss(model,probe,device)
                        charge_probe(ledger,ps.probe_micro*micro,tv.seconds);losses[str(chosen)]=lv
                    else:lv={0.0:l0,mid:lm,ps.amax:l1}[chosen]
                    accept=chosen>0 and lv<=l0-ps.margin
                if accept:
                    spec.assign_parameters(current+chosen*direction)
                    if ps.selector in {"all","*"}:
                        state_ops=transport_optimizer_state(opt,ps.horizon,ps.state)
                    else:
                        params=selected_params(spec,infos,ps.selector,n_layers)
                        state_ops=transport_selected_state(opt,params,ps.state)
                    ledger.add_parameter_ops(DIRECT_ASSIGN_OPS_PER_PARAMETER*spec.numel+state_ops,charged=True)
                    ledger.synthetic_steps+=1;ledger.optimizer_steps+=1;actions+=1;accepted+=1;exact_since=0
                    if ps.clock=="advance":cursor+=skip;virtual+=ps.horizon;skipped_tokens+=skip*micro
                    events.append({"kind":"accept","F":ledger.charged_flop_equiv,"virtual":virtual,"alpha":chosen,
                                   "horizon":ps.horizon,"probe_losses":losses,"direction_norm":float(direction.norm()),**meta})
                    if ledger.charged_flop_equiv>=next_eval:record()
                    continue
                spec.assign_parameters(current);actions+=1;rejected+=1;exact_since=0
                events.append({"kind":"reject","F":ledger.charged_flop_equiv,"virtual":virtual,"probe_losses":losses,"chosen_alpha":chosen,**meta})
            except Exception as exc:
                spec.assign_parameters(current);actions+=1;rejected+=1;exact_since=0
                events.append({"kind":"probe_failure","error":f"{type(exc).__name__}: {exc}","F":ledger.charged_flop_equiv})

        # Exact fallback / ordinary exact arm.
        k=int(args.grad_accum)
        if cursor+k>len(stream):termination="stream_exhausted";break
        batches=stream[cursor:cursor+k];cursor+=k
        before=spec.flatten_parameters(dtype=torch.float32)
        with Timer() as tt:
            _g,loss,_,_=exact_gradient(model,opt,spec,batches,device,need_forward_features=False,token_sample_max=args.token_sample_max)
            opt.step();opt.zero_grad(set_to_none=True)
        ledger.add_exact(full,k,tt.seconds,charged=True);ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER*spec.numel,charged=True)
        ledger.optimizer_steps+=1;virtual+=1;exact_since+=1
        after=spec.flatten_parameters(dtype=torch.float32);update=after-before;update_history.append(update)
        exact_weights.append(after.cpu().to(_dtype_from_name(args.history_dtype)))
        events.append({"kind":"exact","F":ledger.charged_flop_equiv,"virtual":virtual,"train_loss":loss,
                       "update_norm":float(update.norm())})
        if ledger.charged_flop_equiv>=next_eval:record()
    if not trace or trace[-1]["charged_flop_equiv"]!=ledger.charged_flop_equiv:record()
    runtime.out.mkdir(parents=True,exist_ok=True)
    result={"arm":runtime.arm,"arm_slug":arm_slug(runtime.arm),"optimizer":pack["build_args"].get("optimizer"),
            "prefix_steps":int(pack["prefix_steps"]),"runtime_args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(runtime).items()},
            "build_args":pack["build_args"],"trace":trace,"events":events,"final_ledger":ledger.as_dict(),
            "virtual_steps":virtual,"jump_actions":actions,"accepted":accepted,"rejected":rejected,
            "acceptance_rate":accepted/max(actions,1),"skipped_tokens":skipped_tokens,"termination":termination}
    path=runtime.out/"result.json";path.write_text(json.dumps(result,indent=2,allow_nan=True));print('[result] wrote',path);return path


def selftest():
    for arm in ["vf:mean:8:4:8:0.25:preserve:advance:1:0.0:all","qf:dmd:4:4:8:0.25:preserve:hold:2:0.001:muon"]:
        p=parse_probe(arm);assert p.horizon==4 and p.probe_micro>0
    a=fitted_alpha(1.0,.8,.9,1.0);assert 0<=a<=1
    print('[selftest] forward-probe fast-forward PASS')


def parser():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);sub.add_parser('selftest')
    bp=sub.add_parser('branch');bp.add_argument('--pack',type=Path,required=True);bp.add_argument('--arm',required=True);bp.add_argument('--out',type=Path,required=True)
    bp.add_argument('--device',default='cuda');bp.add_argument('--history-device',default=None);bp.add_argument('--history-dtype',default=None)
    bp.add_argument('--compute-budget',type=float,default=1800);bp.add_argument('--base-lr-scale',type=float,default=0.30);bp.add_argument('--jump-start-equiv',type=float,default=100);bp.add_argument('--jump-stop-equiv',type=float,default=1200)
    bp.add_argument('--eval-every-equiv',type=float,default=100);bp.add_argument('--warmdown-ratio',type=float,default=None);bp.add_argument('--final-lr-frac',type=float,default=None)
    bp.add_argument('--lawa-window',type=int,default=225);bp.add_argument('--seed',type=int,default=None)
    return ap

def main():
    a=parser().parse_args()
    if a.command=='selftest':selftest();return
    _install_canonical_build();run(a)
if __name__=='__main__':main()
