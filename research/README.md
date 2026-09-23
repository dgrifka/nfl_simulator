# research/

The numbered build and exploratory scripts behind the **previous** model (v1),
which is retired. They are kept because `docs/research/00`–`74` cite them as
their evidence, and a decision record whose evidence has been deleted is worth
nothing.

**The v1 research scripts run at tag v1.4.3**, against the package as it stood
there. They do not run against this version: the modules they import
(`simulator`, `components`, `fg_model`, `ledger`, `render` and the rest) were
removed when v1 was retired.

The model of record has no scripts here. Its fit is a library entry point:

```bash
uv run python -m nfl_simulator.process_meter.fit --data-dir data --out /tmp/w.json
```
