#!/usr/bin/env python3
"""Merge and plot Muon-aware averaging trajectory results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def load_results(root: Path) -> list[dict[str, Any]]:
    out=[]
    for p in root.rglob("result.json"):
        try:
            r=json.loads(p.read_text()); r["result_path"]=str(p); out.append(r)
        except Exception as exc:
            print("skip", p, exc)
    if not out:
        raise SystemExit(f"no result.json under {root}")
    return out


def paired_delta(row: dict[str, Any]) -> float:
    return float(row.get("paired_delta_vs_raw", float("nan")))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    results=load_results(args.root)
    final=[]
    for r in results:
        evals=r.get("recipe_evaluations", [])
        if not evals: continue
        step=max(int(x["step"]) for x in evals)
        for x in evals:
            if int(x["step"]) != step: continue
            final.append({
                "trajectory_id":r["trajectory_id"],
                "optimizer":r.get("optimizer"),
                "recipe":x["recipe"],
                "final_bpb":x["bpb"],
                "final_bpb_se":x.get("bpb_se"),
                "paired_delta_vs_raw":x.get("paired_delta_vs_raw",0.0),
                "paired_se_vs_raw":x.get("paired_se_vs_raw",0.0),
                "paired_z_vs_raw":x.get("paired_z_vs_raw",0.0),
                "mean_alpha_by_numel":x.get("mean_alpha_by_numel"),
                "displacement_norm":x.get("displacement_norm"),
                "charged_flop_equiv":x.get("charged_flop_equiv"),
                "result_path":r["result_path"],
            })
    final.sort(key=lambda x: float(x["final_bpb"]))
    fields=sorted({k for r in final for k in r})
    with (args.out/"leaderboard.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(final)

    # Best recipe per family/trajectory.
    print(f'{"trajectory":24} {"recipe":42} {"bpb":>10} {"delta raw":>11} {"z":>9}')
    print("-"*104)
    for row in final[:40]:
        print(f'{row["trajectory_id"][:24]:24} {row["recipe"][:42]:42} {float(row["final_bpb"]):10.6f} '
              f'{float(row["paired_delta_vs_raw"]):+11.6f} {float(row["paired_z_vs_raw"]):+9.2f}')

    # Figure: final delta by recipe for the best trajectory.
    if final:
        best_traj=final[0]["trajectory_id"]
        rows=[r for r in final if r["trajectory_id"]==best_traj and r["recipe"]!="raw"]
        rows=sorted(rows,key=lambda x:float(x["paired_delta_vs_raw"]))[:20]
        fig,ax=plt.subplots(figsize=(10,max(4,0.32*len(rows))))
        y=np.arange(len(rows))
        vals=np.array([float(r["paired_delta_vs_raw"]) for r in rows])
        err=2*np.array([float(r["paired_se_vs_raw"]) for r in rows])
        ax.barh(y,vals,xerr=err)
        ax.axvline(0,linewidth=1)
        ax.set_yticks(y,[r["recipe"] for r in rows]); ax.invert_yaxis()
        ax.set_xlabel("BPB delta vs raw endpoint (lower is better)")
        ax.set_title(f"Muon-aware averaging mechanisms — {best_traj}")
        fig.tight_layout(); fig.savefig(args.out/"fig_recipe_delta.png",dpi=180); fig.savefig(args.out/"fig_recipe_delta.pdf")
        plt.close(fig)

    # Figure: group noise fraction vs marginal group averaging gain.
    points=[]
    by_traj={r["trajectory_id"]:r for r in results}
    for row in final:
        if not row["recipe"].startswith("group:"): continue
        parts=row["recipe"].split(":"); selector=parts[1]; key=f'{parts[2]}x{parts[3]}'
        rr=by_traj[row["trajectory_id"]]
        # aggregate matching group from final tensor stats
        stat_keys=[k for k in rr.get("tensor_stats",{}) if k.endswith("_"+key)]
        if not stat_keys: continue
        stats=rr["tensor_stats"][sorted(stat_keys)[-1]]
        matches=[]
        for s in stats:
            role=s.get("role"); kind=s.get("optimizer_kind")
            ok=(selector=="all" or selector==role or selector==kind or
                selector=="muon_attention" and kind=="muon" and role=="attention" or
                selector=="muon_mlp" and kind=="muon" and role=="mlp")
            if ok: matches.append(s)
        if not matches: continue
        w=np.array([max(float(s["numel"]),1) for s in matches]); nf=np.array([float(s["noise_fraction"]) for s in matches])
        noise=float(np.average(nf,weights=w))
        points.append((noise,float(row["paired_delta_vs_raw"]),selector,row["trajectory_id"]))
    if points:
        fig,ax=plt.subplots(figsize=(8,6))
        for noise,delta,label,traj in points:
            ax.scatter(noise,-delta)
            ax.annotate(f"{traj}:{label}",(noise,-delta),fontsize=7,xytext=(3,3),textcoords="offset points")
        ax.set_xlabel("causal update noise fraction")
        ax.set_ylabel("averaging improvement (-Δ BPB)")
        ax.set_title("Does noisy/oscillatory geometry predict averaging benefit?")
        fig.tight_layout(); fig.savefig(args.out/"fig_noise_vs_gain.png",dpi=180); fig.savefig(args.out/"fig_noise_vs_gain.pdf")
        plt.close(fig)

    digest=[]
    for traj in sorted({r["trajectory_id"] for r in final}):
        rows=[r for r in final if r["trajectory_id"]==traj]
        raw=next((r for r in rows if r["recipe"]=="raw"),None)
        best=min(rows,key=lambda x:float(x["final_bpb"]))
        digest.append(f"{traj}: raw={float(raw['final_bpb']):.6f} best={best['recipe']} {float(best['final_bpb']):.6f} "
                      f"gain={-float(best.get('paired_delta_vs_raw',0)):.6f}")
    (args.out/"DIGEST.txt").write_text("\n".join(digest)+"\n")
    print("wrote", args.out)


if __name__=="__main__": main()
