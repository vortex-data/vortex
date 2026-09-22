
<details>
<summary>:warning: <b>Unknown Walltime execution environment detected</b></summary>

> Using the Walltime instrument on standard Hosted Runners will lead to inconsistent data.
>
> For the most accurate results, we recommend using [CodSpeed Macro Runners](https://codspeed.io/docs/instruments/walltime): bare-metal machines fine-tuned for performance measurement consistency.

</details>
<details>
<summary>:warning: <b>Different runtime environments detected</b></summary>

> Some benchmarks with significant performance changes were compared across different runtime environments,
> which may affect the accuracy of the results.
>
> [Open the report in CodSpeed to investigate](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=comparison_issues)

</details>

`⚡ 20` improved benchmarks  
`❌ 30` regressed benchmarks  
`✅ 2302` untouched benchmarks  
`⏩ 293` skipped benchmarks[^skipped]  


> [!WARNING]
> Please fix the performance issues or [acknowledge them on CodSpeed](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Aregression&utm_source=github&utm_medium=check&utm_content=acknowledge).

### Performance Changes

|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |
| --- | ---- | --------- | ------ | ------ | ---------- |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 4.1 µs | 44 µs | -90.59% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantLhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantLhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 4.1 µs | 42.2 µs | -90.16% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (Columns, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28Columns%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 4.8 µs | 42.3 µs | -88.66% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (Columns, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28Columns%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 5.4 µs | 43.3 µs | -87.42% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 11 µs | 51.4 µs | -78.57% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantLhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantLhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 11 µs | 49.2 µs | -77.54% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (Columns, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28Columns%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 11.7 µs | 49.3 µs | -76.27% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 14 µs | 44.7 µs | -68.66% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (Columns, NullOnlyFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28Columns%2C+NullOnlyFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 34.4 µs | 72.5 µs | -52.61% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, NullOnlyFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+NullOnlyFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 42.9 µs | 74.5 µs | -42.41% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 27.5 µs | 47.3 µs | -41.73% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 26 µs | 44.5 µs | -41.61% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 25.9 µs | 43 µs | -39.77% |
| ❌ | WallTime | [`` multiversioned_avx512[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 25.9 µs | 43 µs | -39.62% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 27.8 µs | 44.3 µs | -37.25% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 27.8 µs | 42.3 µs | -34.33% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 34.5 µs | 51.4 µs | -32.88% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 29.3 µs | 43.5 µs | -32.64% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 33.4 µs | 49.4 µs | -32.43% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 34.8 µs | 49.5 µs | -29.76% |
| ... | ... | ... | ... | ... | ... |

<br/>

> :information_source: _Only the first 20 benchmarks are displayed. [Go to the app to view all benchmarks](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=all_benchmarks)._

> [!TIP]
> Investigate this regression by commenting `@codspeedbot fix this regression` on this PR, or directly use the [CodSpeed MCP](https://codspeed.io/docs/ai/mcp?utm_source=github&utm_medium=check&utm_content=mcp_tip_regression) with your agent.

---

<sub>Comparing <code>ct/row-fn-bool-retry</code> (31c9d94) with <code>ct/row-fn-bool-bench</code> (a01de34)</sub>

<a href="https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=button">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://codspeed.io/pr-report/open-in-codspeed-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://codspeed.io/pr-report/open-in-codspeed-light.svg">
    <img alt="Open in CodSpeed" src="https://codspeed.io/pr-report/open-in-codspeed-light.svg" width="169" height="32">
  </picture>
</a>


[^skipped]: 293 benchmarks were skipped, so the baseline results were used instead. If they were deleted from the codebase, [click here and archive them to remove them from the performance reports](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Askipped&utm_source=github&utm_medium=check&utm_content=archive).

