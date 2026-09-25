#!/usr/bin/env bash
# Run inside a server container (root): where each scheduler process may run, where its
# threads are running now, and on which NUMA node its resident memory lives.
node_of_cpu() { if (( $1 < 72 )); then echo 0; else echo 1; fi; }
for pid in $(pgrep -f '^sglang::scheduler'); do
  name=$(tr -d '\0' < /proc/$pid/cmdline | cut -c1-40)
  cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/$pid/status)
  mems=$(awk '/Mems_allowed_list/ {print $2}' /proc/$pid/status)
  n0=0; n1=0
  for psr in $(ps -L -o psr= -p "$pid"); do
    if [[ $(node_of_cpu "$psr") == 0 ]]; then n0=$((n0 + 1)); else n1=$((n1 + 1)); fi
  done
  pages=$(awk '{for (i = 1; i <= NF; i++) if ($i ~ /^N[0-9]+=/) {split($i, a, "="); s[a[1]] += a[2]}}
               END {for (k in s) printf "%s=%d ", k, s[k]}' /proc/$pid/numa_maps 2>/dev/null)
  policy=$(awk 'NR == 1 {print $2}' /proc/$pid/numa_maps 2>/dev/null)
  echo "$name pid=$pid cpus_allowed=$cpus mems_allowed=$mems threads_on_node0=$n0 threads_on_node1=$n1 pages[$pages] first_mapping_policy=$policy"
done
