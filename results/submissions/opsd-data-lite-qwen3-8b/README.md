# OPSD-data-lite on Qwen3-8B


This submission records the aggregate table supplied for the
[OPSD-data-lite](../../../rsi/methods/opsd-data-lite/). It is included so
the result is represented in the same machine-readable structure as other
DataLite-RSI modules. 

## Headline

Under the OPSD math competition suite, the reported aggressive third iteration
reaches **62.1** after 100 steps on 32 selected examples, compared with **58.7**
for the base model and **60.3** for the matched 100-step full-pool OPSD control.
This is a reported gain of **+3.4** points over base and **+1.8** points over
matched-step OPSD. Every data-lite iteration uses the same 100-step budget;
only the selected subset shrinks (2,400 to 1,600 to 32).

| Setting | Steps | Samples | AIME25 | HMMT25 | AIME26 | HMMT26 | BRUMO25 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | 0 | -- | 66.7 | 44.2 | 67.8 | 45.5 | 69.2 | 58.7 |
| OPSD | 50 | full pool | 69.7 | 46.1 | 69.4 | 46.4 | 71.4 | 60.6 |
| OPSD | 75 | full pool | 68.1 | 44.7 | 68.6 | 46.0 | 70.3 | 59.5 |
| OPSD | 100 | full pool | 69.2 | 45.6 | 68.9 | 46.5 | 71.4 | 60.3 |
| OPSD-data-lite, iter 1 | 100 | 2,400 | 71.4 | 45.6 | 70.3 | 45.5 | 70.6 | 60.7 |
| OPSD-data-lite, iter 2 | 100 | 1,600 | 71.7 | 46.4 | 71.9 | 47.2 | 72.5 | 61.9 |
| **OPSD-data-lite, aggressive iter 3** | 100 | 32 | 71.4 | 46.7 | 72.5 | 47.2 | 72.7 | **62.1** |

Scores are reported percentages, averaged over three evaluation seeds. The
protocol uses Qwen3 thinking mode, temperature 1.0, top-p 0.95, top-k -1, and
12 samples per problem. The suite score is an unweighted mean over five
datasets.

