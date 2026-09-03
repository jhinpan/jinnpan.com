# GLM-5.3 MXFP4 Terminal-Bench 4.0 live board

This directory documents the public data contract for the live companion to
Experiment 004. The dashboard is served by jinnpan.com at
`/sources/glm53-mxfp4-tb4-live.html`.

## Data path

The evaluation host derives one sanitized JSON snapshot from Harbor job
results, endpoint health, and the valid-attempt ledger. It updates the public
Gist once per minute:

- Gist: <https://gist.github.com/jhinpan/e73f4d28e91332c8524f03a682923a1d>
- Raw feed:
  <https://gist.githubusercontent.com/jhinpan/e73f4d28e91332c8524f03a682923a1d/raw/tb4-status.json>

The static dashboard fetches that feed every 30 seconds with cache busting.
The page marks data older than three minutes as stale. A failed fetch keeps the
last successful snapshot visible and does not pretend it is current.

## Counting contract

The fixed target is 63 CPU tasks times five attempts, or 315 scored attempts.
An attempt advances the public counter only when a verifier returns a reward.
A zero reward is still a scored model outcome. Infrastructure and verifier
failures without a reward remain in the issue ledger and are replayed from the
narrowest durable boundary.

The board reports:

- scored and remaining attempts;
- tasks with all five scored attempts;
- configured trial concurrency and per-pool progress;
- current request and queue pressure;
- observed terminal-trial rate and a rate-based main-run ETA;
- sanitized issue categories and their resolution state.

## Public boundary

The feed intentionally excludes task prompts, agent trajectories, request
content, internal addresses, hostnames, credentials, filesystem paths, raw
logs, and unpublished benchmark artifacts. The publisher rejects snapshots
containing those classes of data before any external write.

The Gist is a transport for changing aggregate state, not the durable
experiment record. The final, validated results will be archived in this site
repository after every task reaches five scored attempts.
