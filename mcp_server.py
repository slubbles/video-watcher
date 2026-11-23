"""MCP server wrapping motion_judge. Exposes analyze_video(file_path) -> JSON report + markdown."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from motion_judge import analyze, write_report_md

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP

mcp = FastMCP("video-watcher")

@mcp.tool()
def analyze_video(file_path: str) -> dict:
    rep = analyze(Path(file_path))
    out = Path(file_path).with_suffix(".motion.json")
    out.write_text(json.dumps(rep, indent=2))
    md = out.with_suffix(".md")
    write_report_md(rep, md, Path(file_path))
    return {"report": str(out), "markdown": str(md), "segments": rep["segments"], "cuts_s": rep["scene_cuts_s"]}

if __name__ == "__main__":
    mcp.run()
