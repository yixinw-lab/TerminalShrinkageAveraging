# Record attempt v1.1 hotfix

This hotfix changes no scientific settings. It fixes the PR-830 patcher so the
training-loop hook is anchored to semantic markers (`# Go!`, the state-update
`step += 1`, and the post-loop stats marker) rather than an exact neighboring
print string. It also adds `tools/finish_existing_checkout.sh` so an already
cloned/`uv sync`ed checkout can be repaired without reinstalling dependencies.
