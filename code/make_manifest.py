# -*- coding: utf-8 -*-
"""make_manifest.py — 生成结果文件清单（完整 64 位 SHA256）。

- 扫描 <repo>/outputs 下全部文件；
- 清单文件自身（_files_manifest.txt）不列入，避免自引用导致的旧大小/hash；
- 输出：outputs/_files_manifest.txt（每行：相对路径|字节|mtime|sha256:<完整64位>）。

用法：python make_manifest.py --repo D:/develop/Github/DeepSeek_Cup
"""
import argparse
import hashlib
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    out_root = args.repo / "outputs"
    if not out_root.exists():
        raise SystemExit(f"目录不存在: {out_root}")
    rows = []
    for p in sorted(out_root.rglob("*")):
        if not p.is_file() or p.name == "_files_manifest.txt":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = p.relative_to(args.repo).as_posix()
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        rows.append(f"{rel}|{p.stat().st_size}|{mtime}|sha256:{h}")
    (out_root / "_files_manifest.txt").write_text(
        "# files manifest (full SHA256; this file itself excluded)\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {out_root / '_files_manifest.txt'}")


if __name__ == "__main__":
    main()

