# Community release plan

The user’s current objective (2026-10-29) is a clear, faithful repository that
can be shared promptly. Full-budget training, every benchmark experiment and
performance superiority are **not release requirements**. The earlier research
plan is preserved in [the archive](EXPERIMENT_PLAN_ARCHIVE_20261029.md), rather
than being treated as the current acceptance gate.

## Required software evidence

1. Preserve past-only local intention inference, future-informed auxiliary
   learning, and the paper’s policy-KL coordination criterion. Document
   mathematical corrections and deliberate implementation differences.
2. Keep actors and critics local; hidden actions and future targets cannot
   enter acting inputs. Preserve episode boundaries and unavailable-action masks.
3. Validate the unit suite and the documented IPPO, soft-DMS and causal-DMS toy
   commands, including saved-source checkpoint evaluation.
4. Build an installable package and verify its execution outside the checkout.
5. Provide a short reader entry point, benchmark setup links, license and
   citation. Label the environment assumptions and all unverified claims.
6. Preserve existing completed and partial experiment evidence without requiring
   further performance searches. Leave the sibling project untouched.

The research campaigns were intentionally stopped to meet this scope. No
performance ranking, complete experiment matrix or held-out-seed result is
inferred from passing software checks. Further benchmarks are optional work.
The software is published to `LiZhYun/DMS`, branch `main`; the original paper
code is retained on `legacy-paper-code`. The current [status](STATUS.md) links
the verified publication revision and GitHub CI result.
