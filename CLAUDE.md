# Pull request completion

For this repository, promptly merge a pull request into `main` when all of the
following are true: it targets `main`, every required CI check has completed
successfully, and GitHub reports it as clean and mergeable. Do not merge drafts,
PRs with pending or failing checks, conflicts, or a different base branch.

After each merge, verify that `main` contains the merge commit, delete the
merged remote branch, and remove its clean local worktree and local branch.
Never delete an active branch or a worktree with uncommitted changes.
