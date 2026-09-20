# Evidence gaps for the first research collection

This list tracks follow-up materials, independently of website publication.
Current aggregate results are preserved as author-reported and unverified.

| Area | Needed next | Current public wording |
| --- | --- | --- |
| OPSD | Selector, row IDs, logs explaining 100 steps / 1 epoch, full revisions, final-round cost | 32 final selected examples from 9,600 candidates; additional search cost |
| Evolver | Diagnostic sample IDs and per-sample predictions, source release | Full-test mean includes samples used for diagnostic feedback; original Math-3K is stronger |
| Self-Improver | Three run IDs, seed clarification for 42/42/44, acceptance/rollback logs | Seven-set transfer mean; final 6K corpus, intermediate 7K; rollback evidence pending |
| Image | Scene snapshot, adapters, seeds, per-sample judge files | Same 1,000-pair budget; synthesis compute not established as equal |
| Video | Pipeline implementation, complete training settings, predictions, matched-data controls | Five base-model comparisons; five described search iterations versus one iteration recorded in result settings |
| All methods | Public artifacts and license decisions | Missing URLs and `NOASSERTION` retained; no implied permission badge |
| Protocol registry | Explicit feedback/evaluation sets, run roles and budget fields | Seven/eight-set math kept separate; pipeline iterations and training runs not conflated |

## Completed engineering fixes

The release adds full test discovery, CLI fixture checks, a Video-MME CLI startup
fix, the documented treatment of truncated math completions, stronger result
validation, and a generated website/README summary. These are software fixes;
they do not change recorded research scores or establish reproduction.

Useful next experiments include matched-budget random or one-shot selection for
OPSD; independent repeats and broader data-budget comparisons for synthesis;
matched-data controls for VideoRSI; untouched evaluation slices for Evolver; and
non-recursive corpus controls for Self-Improver. Their absence should remain
visible alongside the relevant case.
