# third_party - upstream code managed via git subtree

This directory holds a verbatim copy of upstream source code that is tracked
as a **git subtree** inside the kvpress repository.

---

## `instruction_following_eval`

| Field | Value |
|---|---|
| Upstream repo | `https://github.com/google-research/google-research.git` |
| Upstream sub-path | `instruction_following_eval/` |
| Pinned commit | `aa633e5105c702b47a4dd836d9b6eca39984a0fe` |
| Local prefix | `evaluation/benchmarks/ifeval/third_party/instruction_following_eval` |

### Adaptations applied to the upstream code

The files were lightly adapted before being placed here (same changes as the
previous inline copies):

- Replaced `absl.logging` with stdlib `logging`.
- Made `langdetect` an optional dependency (skips language checks when absent).
- Replaced `immutabledict` with plain `dict`.
- Updated intra-package imports to use the `instruction_following_eval.*`
  namespace instead of the old `benchmarks.ifeval.*` namespace.

### Adding the subtree for the first time

Because the upstream repository is very large, the recommended approach is to
add only the `instruction_following_eval` sub-tree using a filtered remote:

```bash
# 1. Add the remote (one-time setup)
git remote add google-research https://github.com/google-research/google-research.git
git fetch google-research

# 2. Use git subtree to graft the sub-directory
git subtree add \
  --prefix=evaluation/benchmarks/ifeval/third_party/instruction_following_eval \
  google-research/master \
  --squash
```

> **Note:** `git subtree add` copies the *entire* repository root into the
> prefix, not just the sub-directory.  If you only want the
> `instruction_following_eval/` sub-tree you can first create a filtered
> branch with `git filter-branch` or `git-filter-repo`, then subtree-add
> from that branch.  The files currently committed here were copied manually
> from commit `aa633e5105c702b47a4dd836d9b6eca39984a0fe`.

### Updating the subtree

```bash
git fetch google-research

git subtree pull \
  --prefix=evaluation/benchmarks/ifeval/third_party/instruction_following_eval \
  google-research/master \
  --squash
```

After pulling, re-apply any adaptations listed above if the upstream files
changed, then commit.
