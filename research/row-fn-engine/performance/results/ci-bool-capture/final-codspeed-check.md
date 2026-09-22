
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

`⚡ 40` improved benchmarks  
`❌ 3` regressed benchmarks  
`✅ 2237` untouched benchmarks  
`⏩ 293` skipped benchmarks[^skipped]  


> [!WARNING]
> Please fix the performance issues or [acknowledge them on CodSpeed](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Aregression&utm_source=github&utm_medium=check&utm_content=acknowledge).

### Performance Changes

|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |
| --- | ---- | --------- | ------ | ------ | ---------- |
| ❌ | WallTime | [`` filtered_sink_i64_avx512[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx512%3A%3Afiltered_sink_i64_avx512%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22.4 µs | 26.4 µs | -15.22% |
| ❌ | Simulation | [`` take_fsl_u32_random[128, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_u32_random%5B128%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 207.1 µs | 236.8 µs | -12.54% |
| ❌ | Simulation | [`` take_fsl_f16_random[256, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_f16_random%5B256%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 203.9 µs | 230.3 µs | -11.46% |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 226.8 µs | 27.8 µs | ×8.2 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 210.6 µs | 28 µs | ×7.5 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 34 µs | ×6.1 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.1 µs | 34 µs | ×6.1 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (Columns, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 220 µs | 36.9 µs | ×6 |
| ⚡ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.8 µs | 36.8 µs | ×5.6 |
| ⚡ | WallTime | [`` multiversioned_avx512[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.3 µs | 36.8 µs | ×5.6 |
| ⚡ | WallTime | [`` plain_avx512[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Aplain_avx512%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 38.6 µs | ×5.3 |
| ⚡ | WallTime | [`` plain_avx512[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Aplain_avx512%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 38.7 µs | ×5.3 |
| ⚡ | WallTime | [`` multiversioned_neon[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Amultiversioned_neon%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 174.4 µs | 51.6 µs | ×3.4 |
| ⚡ | WallTime | [`` plain_neon[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Aplain_neon%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 174.4 µs | 52 µs | ×3.4 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 279.1 µs | 84.6 µs | ×3.3 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 267.6 µs | 84.6 µs | ×3.2 |
| ⚡ | WallTime | [`` plain_neon[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Aplain_neon%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 171.7 µs | 55.1 µs | ×3.1 |
| ⚡ | WallTime | [`` multiversioned_neon[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Amultiversioned_neon%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 171 µs | 55.2 µs | ×3.1 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (Columns, ObservableFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+ObservableFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 277.5 µs | 94.2 µs | ×2.9 |
| ⚡ | WallTime | [`` multiversioned_avx2[16384, (Columns, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 110.4 µs | 43 µs | ×2.6 |
| ... | ... | ... | ... | ... | ... |

<br/>

> :information_source: _Only the first 20 benchmarks are displayed. [Go to the app to view all benchmarks](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=all_benchmarks)._

> [!TIP]
> Investigate this regression by commenting `@codspeedbot fix this regression` on this PR, or directly use the [CodSpeed MCP](https://codspeed.io/docs/ai/mcp?utm_source=github&utm_medium=check&utm_content=mcp_tip_regression) with your agent.

---

<sub>Comparing <code>ct/row-fn-bool-retry</code> (363bac7) with <code>ct/row-fn-bool-bench</code> (2449ffb)</sub>

<a href="https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=button">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://codspeed.io/pr-report/open-in-codspeed-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://codspeed.io/pr-report/open-in-codspeed-light.svg">
    <img alt="Open in CodSpeed" src="https://codspeed.io/pr-report/open-in-codspeed-light.svg" width="169" height="32">
  </picture>
</a>


[^skipped]: 293 benchmarks were skipped, so the baseline results were used instead. If they were deleted from the codebase, [click here and archive them to remove them from the performance reports](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Askipped&utm_source=github&utm_medium=check&utm_content=archive).

