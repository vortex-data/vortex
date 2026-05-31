# Copy-paste prompt to continue the FSSTView work

Paste the block below to a fresh agent session in the Vortex repo.

---

Continue work on the `FSSTView` encoding in the Vortex repo. It's already implemented and
merge-ready on branch `claude/fsstview-array-listview-TdW45` (17 commits ahead of `develop`).
Read `FSSTVIEW_HANDOVER.md` and `encodings/fsst/benches/README.md` first for full context and
benchmark numbers.

Background: `FSSTView` (in `encodings/fsst/src/fsstview/`) is a ListView-style FSST that stores
compressed codes addressed by separate `offsets` + `sizes` arrays, making `filter`/`take`/`slice`
metadata-only (no code-heap rewrite). On real FineWeb data the view wins up to 8.6× on chained
ops over long strings. Its one measured weakness: on highly selective predicates over short
columns it pays a fixed ~130 µs floor because `fsstview_from_fsst` derives the full `sizes` array
(over all rows) even when <1% survive.

Task: eliminate that conversion floor without regressing the cases the view already wins. Approach:

1. Confirm the current behaviour first: run
   `python3 encodings/fsst/benches/fineweb_queries_extract.py` (needs `pip install duckdb`, network
   to HuggingFace), then
   `FINEWEB_DIR=/tmp cargo bench -p vortex-fsst --bench fsst_view_fineweb_queries`. Note the
   `url/vortex`, `url/google_and`, `url/espn_*` rows where `view` trails `fsst`.
2. Implement a cheaper `sizes` representation so a selective filter doesn't materialize sizes for
   discarded rows — e.g. derive `sizes` lazily from `offsets` at canonicalize time, or store it in
   the narrowest int width that fits. `filter`/`take` currently filter a concrete `codes_sizes`
   child array, so whatever you choose must keep those ops metadata-only and still composable
   across a chain (do NOT fuse conversion into filter).
3. Prove it with the same methodology, not instruction counts: samply (set
   `perf_event_paranoid=1`) for wall-clock and the real `fsst_view_fineweb_queries` bench. Show the
   selective `url` queries improve AND the winning cases (`chain text`, `dump_eq`, `date_prefix`)
   do not regress.
4. Keep it merge-clean: `cargo test -p vortex-fsst` (107 tests), `cargo clippy -p vortex-fsst
   --all-targets --all-features`, `cargo +nightly fmt --all`. Add/adjust tests for any new
   representation. Update `benches/README.md` and `FSSTVIEW_HANDOVER.md` with new numbers. Commit
   with sign-off `Signed-off-by: Joe Isaacs <joe.isaacs@live.co.uk>` and push to the same branch.

Be rigorous about measurement: instruction count is not time, and synthetic micro-loops mislead —
always validate on the real FineWeb columns/query masks. If a change doesn't actually help the real
workload, say so and revert it rather than shipping it.
