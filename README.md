# video-watcher
Local motion-quality analyzer for video renders. Pairs with any video-native model: the machine measures cuts, pacing, holds, flow and judder; the model judges taste.

## CLI
```
python3 motion_judge.py render.mp4 --out render.motion.json
```
Outputs JSON report + Markdown table + annotated contact sheet.

## MCP
```
pip install -r requirements.txt
python3 mcp_server.py
```
Exposes `analyze_video(file_path)` returning segment evidence for the agent.

## What it measures
- Scene cuts (clustered, micro-segments merged)
- Per-segment: duration, kind (moving / low-motion / static hold), mean optical flow
- Frozen-frame runs and judder proxies
- Luminance + whole-frame contrast proxy
