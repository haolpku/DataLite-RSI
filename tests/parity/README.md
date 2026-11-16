# Parity suite

Checks that `rsi.framework` reproduces the observable behavior of
`open-dataflow==1.0.10` — the package this runtime replaced. 183 cases across
four groups: storage read/write and type coercion, compile/forward tracing,
batched and streaming pipelines, and serving request/response behavior.

Each case is written once in `cases.py` and evaluated twice, against either
implementation, through the adapters in `adapters.py`. `cases.py` imports
neither implementation directly.

| File | Role |
| --- | --- |
| `cases.py` | The case definitions, implementation-agnostic |
| `adapters.py` | Binds the cases to the reference or to `rsi.framework` |
| `baseline_open_dataflow.json` | Results recorded from the real reference package |
| `record_baseline.py` | Regenerates that recording (a script, not a test) |
| `test_open_dataflow_parity.py` | The `parity`-marked comparison tests |

## Running

The **offline** half needs nothing special and runs by default:

```bash
pytest -q tests/test_dataflow_parity_offline.py
```

It compares `rsi.framework` against the recorded baseline, so the alignment is
checked in the environment the framework actually runs in.

The **reference** half is excluded by default (`-m "not parity"` in
`pytest.ini`) and needs its own environment:

```bash
python -m venv .venv-parity && source .venv-parity/bin/activate
pip install "open-dataflow==1.0.10" pandas requests pytest
pytest -m parity tests/parity
```

It re-derives every case from the real package and asserts the recording still
holds, then compares both implementations in one process. Without it, the
stored baseline could silently drift into an echo of this implementation.

## External checkout requirement

`adapters.py` also loads the DataFlow-Evolver `compat/` fixes, because the
comparison target is *the reference package plus those fixes* — they correct SSE
aggregation, the `reasoning` field alias, the tri-state `enable_thinking`
policy, and pandas missing values leaking out of `FileStorage.read("dict")`.

Those live in a separate read-only checkout that is **not** part of this
repository. Point `DFE_BASELINE_REPO` at it, or place it as a sibling directory
named `DataFlow-Evolver`:

```bash
export DFE_BASELINE_REPO=/path/to/DataFlow-Evolver
```

Without it the parity tests raise a clear `RuntimeError`. The default `pytest`
run never reaches that code, so a contributor who does not have the checkout is
unaffected.

## Re-recording

Only when the reference environment changes on purpose:

```bash
python tests/parity/record_baseline.py
```

Run it in the parity environment. It records the `open-dataflow`, pandas and
Python versions alongside the results; the offline suite skips the
type-inference groups when the local pandas major version differs from the
recording and says so.

See [the parity audit](../../docs/dataflow-serving-parity.md) for what is
verified equivalent and what is not.
