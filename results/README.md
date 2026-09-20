# Result submissions

Browse the [reported research results](../README.md#reported-results) or the
[interactive homepage](https://haolpku.github.io/DataLite-RSI/#rs-results).
The research collection includes unverified submissions; the verified leaderboard
remains a separate view. Read [protocols](../docs/protocol.md) before comparing scores.

Each submission lives in `results/submissions/<submission-id>/` and contains a
`result.json` copied from [`../templates/result.json`](../templates/result.json).

New submissions must use `"status": "unverified"`. Maintainers reproduce or
audit the result and then generate `leaderboard.json`; contributors should not
edit the leaderboard directly.

Small aggregate CSV/JSON files and compact plots may accompany a submission.
Store raw generations, videos, trajectories, checkpoints, and other large
artifacts on Hugging Face and link them from `result.json`.

Baseline and post-RSI measurements must use comparable evaluation settings.
Report all planned runs, including failures and exclusions, and document any
manual selection or intervention.

