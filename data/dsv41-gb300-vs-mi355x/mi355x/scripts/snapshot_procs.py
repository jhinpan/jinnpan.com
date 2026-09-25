#!/usr/bin/env python3
"""Record every process of one server session: name, allowed CPUs, and the
NUMA node of its resident pages (from /proc/PID/numa_maps), as binding proof."""
import json
import re
import sys
from pathlib import Path

session, out = int(sys.argv[1]), Path(sys.argv[2])
rows = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        stat = (proc / "stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        if int(fields[3]) != session:  # field 6 of stat: session id
            continue
        status = dict(line.split(":\t", 1) for line in (proc / "status").read_text().splitlines() if ":\t" in line)
        pages = {}
        for line in (proc / "numa_maps").read_text().splitlines():
            for node, count in re.findall(r"\bN(\d+)=(\d+)", line):
                pages[f"N{node}"] = pages.get(f"N{node}", 0) + int(count)
        rows.append(dict(pid=int(proc.name), name=status.get("Name", "").strip(),
                         cmdline=(proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:160],
                         cpus_allowed=status.get("Cpus_allowed_list", "").strip(),
                         mems_allowed=status.get("Mems_allowed_list", "").strip(),
                         numa_pages=pages))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
out.write_text(json.dumps(sorted(rows, key=lambda r: r["pid"]), indent=2) + "\n")
print(f"{len(rows)} processes recorded")
