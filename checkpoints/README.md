# Policy checkpoints

`all-terrain-v6.txt` is the recommended controller. It was trained in staged
all-terrain passes, always keeping every course in the training suite while
temporarily weighting `GAPS`, `JUMP`, and `SLALOM`. On a disjoint audit of 600
episodes (20 seeds, 10 courses, and 3 evaluation speeds), it completed 94.8%
overall. The per-course completion rates were 100% `FLAT`, 100% `STEPS`, 100%
`RUBBLE`, 93.3% `GAPS`, 85% `MIXED`, 100% `RAMPS`, 90% `SLALOM`, 100% `SLICK`,
98.3% `GAUNTLET`, and 81.7% `JUMP`.

The earlier files are retained as reproducible curriculum stages rather than
as recommended policies. `all-terrain-seed1.txt` predates the finish and jump
controller fixes; `v2` through `v5` are successive hard-course passes.

Every `.txt` file here is inlined into `dist/hexapod-simulator.html` by
`build.sh`, and the dashboard's **Trained policy** panel loads one into the live
simulator — see *Watching a trained policy* in the top-level README. Adding a
checkpoint here is all it takes to make it selectable in the page; the page also
loads a file from disk directly, so a policy that finished training a minute ago
does not need a rebuild.

Re-evaluate it:

```bash
cargo run --release -p hexapod-cli -- eval-all \
  --policy checkpoints/all-terrain-v6.txt --horizon 45 \
  --seed 4000011 --eval-seeds 20
```

Continue a focused pass and write the replacement explicitly:

```bash
cargo run --release -p hexapod-cli -- train-all \
  --resume checkpoints/all-terrain-v6.txt --focus JUMP --iters 200 \
  --output checkpoints/all-terrain-v7.txt
```
