# Demo asset

`docs/assets/loopgepa-demo.gif` is a **live recording** of the real demo — not a
mock. It runs `examples/bugfix-loop/demo.py`, which exercises the actual pipeline
(record traces → diagnose → propose → dry-run promote) against throwaway temp
dirs, offline, with no API key.

## Regenerate

```bash
vhs assets/demo/demo.tape        # requires uv + ttyd + ffmpeg
```

`demo.tape` is checked in so the GIF is reproducible. The recording is honest:
the diagnosis, confidence, and proposed variant come from the code, not a script.
Constraints kept for a good README asset: ≤ 5 MB (currently ~108 KB), font size
18, ~20 s.

## Why it's safe to show as "live"

The demo touches only temporary directories and makes no network calls, so a
recording is deterministic in substance (only the random variant id changes run
to run). We never label a recording "live" unless it actually runs the code.
