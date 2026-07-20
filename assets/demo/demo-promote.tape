# WhyStuck promote GIF — reproducible: `vhs assets/demo/demo-promote.tape` (from repo root).
Output docs/assets/whystuck-promote.gif

Require uv

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 400
Set Padding 24
Set Theme "Catppuccin Mocha"

Hide
Type "clear"
Enter
Show

Sleep 600ms
Type "# approve a fix → WhyStuck edits the file, atomically & reversibly"
Sleep 900ms
Enter
Type "uv run python examples/bugfix-loop/demo_full.py"
Sleep 800ms
Enter
Sleep 6s
