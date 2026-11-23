#!/usr/bin/env python3
"""Local video motion judge: frame-diff, optical flow, scene cuts, frozen frames, judder."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

def analyze(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None; rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))
        if prev_gray is not None:
            diff = np.mean(np.abs(small.astype(np.int16) - prev_gray.reshape(small.shape).astype(np.int16)))
            flow = cv2.calcOpticalFlowFarneback(prev_gray.reshape(small.shape), small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean())
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray.reshape(small.shape).astype(np.float32), small.astype(np.float32))
            rows.append({"t": round(frame_idx/fps, 3), "frame": frame_idx, "diff": round(float(diff), 3),
                         "flow": round(mag, 4), "shift": [round(float(dx),3), round(float(dy),3)],
                         "lum": round(float(small.mean()), 1), "lum_std": round(float(small.std()), 1)})
        prev_gray = small
        frame_idx += 1
    cap.release()
    if not rows: raise SystemExit("no frames analyzed")
    diffs = np.array([r["diff"] for r in rows]); flows = np.array([r["flow"] for r in rows])
    med = float(np.median(diffs))
    raw_cuts = [rows[i]["t"] for i in range(1, len(rows)) if diffs[i] > max(2.0, med*8)]
    # cluster cut hits within 0.4s — transitions span several frames
    cuts = []
    for t in raw_cuts:
        if not cuts or t - cuts[-1] > 0.4:
            cuts.append(t)
    frozen = [rows[i]["t"] for i in range(len(rows)) if diffs[i] < 0.05 and flows[i] < 0.01]
    # judder proxy: consecutive large jumps in flow
    jumps = [rows[i]["t"] for i in range(1, len(rows)) if abs(flows[i]-flows[i-1]) > max(0.5, flows.std()*3)]
    # merge micro-segments (<0.8s) into longer neighbor — transition bleed
    bounds = [0.0] + cuts + [round(total/fps, 3)]
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < 0.8 and b != bounds[-1]:
            continue
        merged.append(b)
    bounds = merged
    segments = []
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        seg = [r for r in rows if s <= r["t"] < e or (i == len(bounds)-2 and r["t"] >= s)]
        if not seg: continue
        seg_flows = np.array([r["flow"] for r in seg]); seg_diffs = np.array([r["diff"] for r in seg])
        still = float(np.mean((seg_diffs < 0.05) & (seg_flows < 0.01)))
        kind = "static_hold" if still > 0.5 else ("moving" if seg_flows.mean() > 0.05 else "low_motion")
        flags = []
        if still > 0.8 and (e - s) > 2.0 and i < len(bounds)-2: flags.append("long_static_suspect")
        cr = float((np.percentile([r["lum"] for r in seg], 90)+1) / (np.percentile([r["lum"] for r in seg], 10)+1))
        if cr < 1.5: flags.append("low_contrast")
        if seg_flows.max() > 2.0: flags.append("flow_spike")
        segs_jumps = [r["t"] for r in seg if r["t"] in jumps]
        if segs_jumps: flags.append("judder_at_" + ",".join(str(x) for x in segs_jumps[:3]))
        segments.append({
            "start_s": round(s, 2), "end_s": round(e, 2), "duration_s": round(e - s, 2),
            "kind": kind, "still_ratio": round(still, 2),
            "flow_mean": round(float(seg_flows.mean()), 4),
            "lum_mean": round(float(np.mean([r["lum"] for r in seg])), 1),
            "lum_std_mean": round(float(np.mean([r["lum_std"] for r in seg])), 1),
            "contrast_ratio": round(float((np.percentile([r["lum"] for r in seg], 90)+1) / (np.percentile([r["lum"] for r in seg], 10)+1)), 2),
            "flags": flags,
        })
    report = {
        "file": str(path), "fps": fps, "width": w, "height": h, "frames": total,
        "duration_s": round(total/fps, 3),
        "diff_mean": round(float(diffs.mean()), 3), "flow_mean": round(float(flows.mean()), 4),
        "flow_max": round(float(flows.max()), 4),
        "scene_cuts_s": cuts,
        "segments": segments,
        "frozen_frames_t": frozen[:50],
        "judder_jump_t": jumps[:50],
        "frames_detail": rows,
    }
    return report

def write_report_md(rep, out_md: Path, video_path: Path):
    lines = [f"# Motion report — {video_path.name}", "",
             f"- {rep['width']}×{rep['height']} @ {rep['fps']}fps, {rep['frames']} frames, {rep['duration_s']}s",
             f"- Cuts at: {', '.join(str(c) for c in rep['scene_cuts_s'])}s", "",
             "| # | start | end | dur | kind | flow | lum | contrast | flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for idx, s in enumerate(rep["segments"], 1):
        lines.append(f"| {idx} | {s['start_s']} | {s['end_s']} | {s['duration_s']} | {s['kind']} | {s['flow_mean']} | {s['lum_mean']} | {s['contrast_ratio']} | {','.join(s['flags'])} |")
    out_md.write_text("\n".join(lines) + "\n")

def make_contact_sheet(video_path: Path, rep, dest: Path):
    tmp = dest.parent / (dest.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    mids = [ (s["start_s"] + s["end_s"]) / 2 for s in rep["segments"] ]
    imgs = []
    import subprocess
    for i, t in enumerate(mids, 1):
        p = tmp / f"seg{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(video_path),
                        "-frames:v", "1", str(p)], check=True)
        img = cv2.imread(str(p))
        imgs.append(cv2.resize(img, (240, 427)))
    cols = min(5, len(imgs))
    rowsn = (len(imgs) + cols - 1) // cols
    while len(imgs) < cols * rowsn: imgs.append(np.zeros_like(imgs[0]))
    grid = imgs[:cols*rowsn]
    sheet = np.vstack([np.hstack(grid[r*cols:(r+1)*cols]) for r in range(rowsn)])
    cv2.imwrite(str(dest), sheet)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = analyze(Path(a.video))
    out = Path(a.out) if a.out else Path(a.video).with_suffix(".motion.json")
    out.write_text(json.dumps(rep, indent=2))
    md = out.with_suffix(".md"); write_report_md(rep, md, Path(a.video))
    sheet = out.with_name(out.stem + ".jpg"); make_contact_sheet(Path(a.video), rep, sheet)
    print(f"frames={rep['frames']} dur={rep['duration_s']}s flow_mean={rep['flow_mean']} segments={len(rep['segments'])} -> {out}, {md}, {sheet}")

if __name__ == "__main__":
    main()
