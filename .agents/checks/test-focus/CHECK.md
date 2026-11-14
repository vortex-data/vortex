+++
id = "test-focus"
title = "Test focus"
description = "Each test checks one behavior, checks it precisely, and cannot pass by accident."
# Tests mostly live in inline `mod tests` inside src files, so select by content, not path. The
# `pub fn test_` arm picks up conformance suites, which have no #[test] of their own.
include = ["**/*.rs"]
content_match = '#\[(test|rstest|tokio::test|hegel::test)\b|^pub fn test_\w+\('
exclude = [
  "**/benches/**",
  "**/generated/**",
  "**/goldenfiles/**",
  "bench-orchestrator/**",
  "benchmarks/**",
  "fuzz/**",
  "java/**",
  "lang/**",
  "vortex-bench/**",
  "vortex-cuda/**",
  "vortex-duckdb/**",
  "vortex-ffi/**",
  "vortex-jni/**",
  "vortex-python/**",
  "vortex-python-abi/**",
  "vortex-python-cuda/**",
  "vortex-sqllogictest/**",
  "vortex-test/**",
  "vortex-tui/**",
  "vortex-web/**",
  "wasm-test/**",
  "xtask/**",
]
max_lines_per_shard = 6000
labels = ["changelog/chore"]
max_findings = 12
allowed_tools = [
  "Bash(cargo test:*)",
  "Bash(RUSTC_WRAPPER= cargo test:*)",
  "Bash(git restore:*)",
]
# The runner checks edits with these after the agent finishes. If one fails, every edit is
# reverted and the findings are reported unfixed.
verify = ["cargo clippy -p {crate} --tests --all-features -- -D warnings"]
+++
# Test focus

You review the Rust tests in the files assigned to you. For each test, decide whether it checks one
behavior, whether it checks that behavior precisely, and whether it could pass when that behavior is
broken. Report what is wrong. Where the fix is clear, local, and does not change what the test
means, make it in the working tree. Your edits become a pull request.

Read `AGENTS.md` (sections "Tests" and "Common Mistakes") and `STYLE.md` ("Testing") before you
start. They take precedence over this file.

## Ground rules

- **Stay in scope.** Review and edit only the files listed under "Assignment". You may read
  other files: the code under test, shared fixtures, guard messages, and conformance suites. Do not
  edit them, and do not report findings in them.
- **Precision over recall.** Report only problems you have confirmed by reading the code. Zero
  findings is a normal, good result. Do not pad the report.
- **Tests only.** Do not change non-test code. If a test exposes a real bug in the code under test,
  report it. Do not fix it.
- **Never weaken coverage.** Do not delete a test, case, or assertion. Do not loosen an assertion
  (for example `assert_eq!` to `assert!(.. .is_ok())`, or `should_panic(expected = ..)` to a bare
  `should_panic`). Do not drop conformance suite entry points or input cases.
- **Never change an expected value unless you derive it independently** from the input contract,
  the documentation, or by hand. A value that the code under test produces is not independent.
  When you cannot derive it, report the finding and do not fix it.
- If a file is too long to review completely, review whole tests in order and list the file in
  `reviewed` only if you finished it. A file left out of `reviewed` is assigned again later.

## How to review a file

For each test function, rstest case set, test macro, and conformance helper:

1. State to yourself the single behavior the name and doc comment claim.
2. Check that the body asserts that behavior, and that the fixture reaches the code path the name
   claims. Read the code under test when you need to (for example, to see whether an encoder falls
   back to another encoding, or which guard rejects an invalid input).
3. Ask how the test would behave if that behavior broke. If it would still pass, that is a finding.
4. Check the body against the categories below.

## Categories

Severity: **high** means the test can pass while the behavior it claims is broken. **medium** means
the test is imprecise or brittle, or covers less than it claims. **low** means maintainability or
style.

| Slug | Look for | Not a problem when | Fix? |
| --- | --- | --- | --- |
| `vacuous-test` (high) | Assertions that never run: `if let Some/Ok(..) { assert.. }` with no `else` where the fixture can make it skip, `let .. else { return }`, loops over empty input, `.ok()` or `let _ =` that drops a `Result` under test, a test with no assertion whose comment claims a result. | The skip is the documented contract (an `Ok`-or-`Err` regression test that only guards against a panic, or a SIMD path with its own engagement test). | Yes, when the expected outcome is obvious: replace the skip with `expect`, `unwrap_err`, or a `match` that asserts the expected variant, or add the assertion that the comment states. |
| `unreached-path` (high) | The fixture never reaches the path the name claims: "multi_frame" data that produces one frame, an encoder that falls back to canonical, a conformance call that gets a canonical array instead of the encoding under test. | The test asserts the precondition (`assert!(x.is::<Enc>())`, a frame count). | Only by adding an assertion that the precondition holds, plus a fixture change if the intent is obvious, and only if the touched test then passes. Otherwise report. |
| `tautological-expected` (high) | The expected value comes from the function under test, or from the same code path (encode, then compare against decode of the same encoded array). | The expected value comes from a separate reference implementation: canonical arrays, a generic accumulator, a hand-built model, as in property tests. | No. Report. |
| `imprecise-failure` (medium) | Invalid-input tests that accept any failure: a bare `#[should_panic]`, `assert!(r.is_err())`, or a panic that a fixture constructor can raise before the call under test. | `should_panic(expected = ..)` or `unwrap_err()` with a substring that names the guard. | Yes: add `expected = "<guard message substring>"` or `unwrap_err()` plus `contains`. Copy the substring from the guard's source. The touched test must pass. |
| `name-mismatch` (medium) | The name or doc comment describes something other than what is asserted, or claims cases ("out of bounds") that the body never covers. | The difference is only wording. | Rename only when the asserted behavior is clearly the intended one. Report doc claims of missing coverage. |
| `multiple-behaviors` (medium) | One test checks several unrelated behaviors (for example compression ratio, roundtrip, and slicing), so a failure does not say what broke and later assertions are hidden. | Several assertions on one operation over related inputs, or setup checks for the main assertion. | Yes, by splitting into tests that share a helper and keep every assertion, when the sections are independent. Otherwise report. |
| `batch-kernel-bypassed` (medium) | A batch kernel (take, filter, compare, cast, and so on) checked only through `execute_scalar` on the lazy result, or only through `len()` or `dtype()` without executing it. | The scalar API is the subject of the test. | Yes: execute the result, or compare it with `assert_arrays_eq!`, which executes its left side. |
| `shared-state` (medium) | Dependence on other tests: static atomics or mutexes that several tests touch, environment variables, fixed temp paths, ordering, timing sleeps. | One test owns the state and a comment says so. | No. Report. |
| `incidental-detail` (low) | Assertions on `Debug` or `Display` text, full error strings, literal encoding-id strings, exact byte sizes, or call counts, when the test is about something else. | That detail is the subject (formatting tests, file-size goldens, "stays encoded" tests). | Rarely. Report. |
| `element-wise-comparison` (low) | Loops or index-by-index `assert_eq!` over `execute_scalar`, `as_slice()[i]`, or validity bits, where `assert_arrays_eq!` against an expected array would do. | dtype-generic helpers over dtypes that `assert_arrays_eq!` does not support (for example unions), or the per-element API is the subject. | Yes: build the expected array with matching dtype and nullability (the macro checks dtype exactly), and put the result under test on the left. |
| `canonicalize-for-assertion` (low) | `.execute::<Canonical>()`, `.execute::<PrimitiveArray>()`, and similar calls whose result only feeds `assert_arrays_eq!`. | The execution is the behavior under test, it runs inside a trace closure, or the canonical value is used for physical, resource, or error assertions. | Yes: remove the execution and pass the lazy array. |
| `copy-paste-cases` (low) | Three or more tests whose bodies differ only in literals or types. | The tests differ in setup that rstest would obscure. | Yes: merge them into one `#[rstest]` with named `#[case::..]`, keeping every input. |
| `test-style` (low) | `unwrap()` where the test could return `VortexResult<()>`, modules named `test` instead of `tests`, `vortex_ensure!(a == b)`. | | Only inside tests you are already editing. Never a finding of its own. |

## Making fixes

- Do not run `cargo fmt`; the runner formats edited files with the pinned nightly.
- Keep each fix small and inside the test code of an assigned file. Follow the code around it:
  imports at module level, the existing `SESSION` and ctx helpers, and rstest case naming.
- This check asks you to verify every edit. After you edit a crate's tests, run:
  1. The runner's verify command, listed in the rules below. If it fails after you finish, the
     runner discards every edit in the shard, so it must pass. If it fails before any of your
     edits, revert them all and report every finding unfixed.
  2. `cargo test -p <crate> --all-features <name>` for each touched test.
  If cargo fails with exactly `sccache: error: Operation not permitted`, rerun with `RUSTC_WRAPPER=`
  in front.
- If clippy or the touched tests fail, revert that fix (`git restore -- <file>`, then reapply your
  other fixes in the file) and report the finding with `"fixed": false`. If a stronger assertion
  fails, the code may have a real bug. Report it as high and include the failure output.
- If you cannot run the checks (a missing toolchain, a timeout), revert your edits in that crate,
  report the findings unfixed, and say so in the summary.

## Naming findings

`item` is the test's path inside the file: the module path plus the function name, such as
`tests::roaring_tests::test_roaring_include_basic`. For a `copy-paste-cases` group, use the
alphabetically first function name; for a conformance helper, the helper's name. A test that you
rename keeps the `item` of its old name. Report `line` as the line of the test's `fn` after your
edits.

## Summary

`summary` is for reviewers who have not seen this prompt: in a sentence or two, what you changed
and why each change keeps the test's meaning, then the exact commands you ran with their results,
including any you could not run and why. No preamble, and no restating of this rubric.
