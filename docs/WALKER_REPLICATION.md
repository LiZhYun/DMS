# Walker development replication

The seed-101 Walker3x2 history-32 IPPO control reached a mean return of
2,322.92, while its residual no-input ablation reached 524.26. Both were trained
for 3M transitions on H100 and evaluated over seeds 400101–400200. The no-input
ablation retains auxiliary training; it is not an exact replay of IPPO because
auxiliary sampling and floating-point clipping arithmetic change the training
trajectory. [Diagnostic](../reports/residual-no-input-rng-diagnostic.json).

The next comparison changes only the development training seed to **103** and
retains three settings:

| Setting | Intention inputs | Auxiliary training | Configuration |
| --- | --- | --- | --- |
| IPPO | None | None | [IPPO](../configs/development/walker-3x2-ippo-fusion-3m-h100-103.json) |
| Residual DMS | Enabled after 100 completed episodes | Enabled | [DMS](../configs/development/walker-3x2-residual-fusion-3m-h100-103.json) |
| Residual no-input control | Disabled throughout training | Enabled | [No-input](../configs/development/walker-3x2-no-input-fusion-3m-h100-103.json) |

All three use the same frozen source, native Walker2d-v2 compatibility
environment, local observation history 32, four environments, H100 GPU family,
and fixed 3M budget. Each checkpoint receives a separate 100-episode CPU
evaluation using seeds 400101–400200 with its saved source. Evaluation performs
no optimization. No held-out training seeds are used and all outcomes are kept.

The questions are whether the strong IPPO result repeats, whether inferred
inputs consistently hurt this setting, and whether the ranking changes with
the training seed. Two development seeds cannot establish final robustness.
The 1,100 paper reference is a full-budget target; this shorter experiment
does not replace the registered final 10M evaluation.

The seed-101 IPPO source predates the optional residual and joint-training
features. The new comparison freezes the current source for all three arms
and records the historical differences in `train.py`, `policy.py`, and
`intention.py`; it does not claim a bytewise rerun of the older implementation.
The additional configuration fields are explicit: residual fusion, zero prior
RL learning rate, and the disabled SMAC terminal-timeout option. IPPO has no
intention modules. The two seed-103 DMS settings differ only in the input
warmup and output directory; IPPO and enabled DMS differ only in algorithm and
output directory.

The [campaign ledger](../reports/development-walker-residual-3m-seed103.json)
records the frozen configurations, source hashes and submission state.
[Native validation](../reports/walker-seed103-native-validation.json), job
**21284426**, passed on A100 in 3m01s. Each setting completed 8,000 transitions
and two saved-source evaluation episodes with unchanged state and no optimizer
steps during evaluation. Enabled DMS had active peer gates; the no-input
control kept zero gates while training its auxiliary models. IPPO had neither
peer inputs nor auxiliary updates. The submission checked nine evidence files.

| Setting | H100 training job | Fixed 100-episode evaluation job |
| --- | --- | --- |
| IPPO | 21284499 | 21284500 |
| Residual DMS | 21284501 | 21284502 |
| Residual no-input control | 21284503 | 21284504 |

Four dependent collectors record the results. Full training and evaluation
results remain pending; the short native check establishes execution only.
