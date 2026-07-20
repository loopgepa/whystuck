# WhyStuck demo GIF — reproducible: `vhs assets/demo/demo.tape` (run from repo root).
# Requires uv + ttyd + ffmpeg. Output is committed at docs/assets/whystuck-demo.gif.
Output docs/assets/whystuck-demo.gif

Require uv

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 420
Set Padding 24
Set Theme "Catppuccin Mocha"

Hide
Type "clear"
Enter
Show

Sleep 600ms
Type "# WhyStuck — why does my bugfix loop keep failing? (local, no API key)"
Sleep 900ms
Enter
Type "uv run python examples/bugfix-loop/demo.py"
Sleep 800ms
Enter
Sleep 6s
