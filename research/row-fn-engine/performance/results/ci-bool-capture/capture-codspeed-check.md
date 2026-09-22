
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

`⚡ 28` improved benchmarks  
`❌ 9` regressed benchmarks  
`✅ 2243` untouched benchmarks  
`⏩ 293` skipped benchmarks[^skipped]  


> [!WARNING]
> Please fix the performance issues or [acknowledge them on CodSpeed](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Aregression&utm_source=github&utm_medium=check&utm_content=acknowledge).

### Performance Changes

|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |
| --- | ---- | --------- | ------ | ------ | ---------- |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 110.4 µs | 231.9 µs | -52.38% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (Columns, NullOnlyFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28Columns%2C+NullOnlyFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 344.4 µs | 468.8 µs | -26.55% |
| ❌ | WallTime | [`` filtered_sink_i64_avx2[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx2%3A%3Afiltered_sink_i64_avx2%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 21.9 µs | 26.1 µs | -15.85% |
| ❌ | WallTime | [`` filtered_owned_i64_avx512[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx512%3A%3Afiltered_owned_i64_avx512%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22.3 µs | 26.2 µs | -14.85% |
| ❌ | WallTime | [`` filtered_owned_i64_avx2[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx2%3A%3Afiltered_owned_i64_avx2%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22 µs | 25.8 µs | -14.76% |
| ❌ | Simulation | [`` take_fsl_f16_random[256, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_f16_random%5B256%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 203.9 µs | 230.1 µs | -11.39% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.1 µs | 231.8 µs | -11.08% |
| ❌ | Simulation | [`` allocate_drop_arrow[65536] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-buffer%2Fbenches%2Fallocation.rs%3A%3Aallocate_drop_arrow%5B65536%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 90.7 µs | 101.8 µs | -10.95% |
| ❌ | WallTime | [`` multiversioned_avx2[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Amultiversioned_avx2%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 231.6 µs | -10.86% |
| ⚡ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.8 µs | 36.9 µs | ×5.6 |
| ⚡ | WallTime | [`` multiversioned_avx512[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.3 µs | 36.9 µs | ×5.6 |
| ⚡ | WallTime | [`` plain_avx512[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Aplain_avx512%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 38.7 µs | ×5.3 |
| ⚡ | WallTime | [`` plain_avx512[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Aplain_avx512%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 38.8 µs | ×5.3 |
| ⚡ | WallTime | [`` multiversioned_neon[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Amultiversioned_neon%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 174.4 µs | 50.8 µs | ×3.4 |
| ⚡ | WallTime | [`` plain_neon[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Aplain_neon%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 174.4 µs | 51.2 µs | ×3.4 |
| ⚡ | WallTime | [`` plain_neon[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Aplain_neon%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 171.7 µs | 54.6 µs | ×3.1 |
| ⚡ | WallTime | [`` multiversioned_neon[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aneon%3A%3Amultiversioned_neon%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 171 µs | 54.6 µs | ×3.1 |
| ⚡ | WallTime | [`` plain_avx2[16384, (ConstantLhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Aplain_avx2%5B16384%2C+%28ConstantLhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.5 µs | 104.2 µs | +98.07% |
| ⚡ | WallTime | [`` plain_avx2[16384, (ConstantRhs, PartialAccepted)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx2%3A%3Aplain_avx2%5B16384%2C+%28ConstantRhs%2C+PartialAccepted%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 206.1 µs | 104.3 µs | +97.57% |
| ⚡ | WallTime | [`` multiversioned_avx512[16384, (ConstantRhs, NullOnlyFailure)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?uri=vortex-array%2Fbenches%2Frow_fn_bool_retry.rs%3A%3Aavx512%3A%3Amultiversioned_avx512%5B16384%2C+%28ConstantRhs%2C+NullOnlyFailure%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 455.1 µs | 286.5 µs | +58.85% |
| ... | ... | ... | ... | ... | ... |

<br/>

> :information_source: _Only the first 20 benchmarks are displayed. [Go to the app to view all benchmarks](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=all_benchmarks)._

> [!TIP]
> Investigate this regression by commenting `@codspeedbot fix this regression` on this PR, or directly use the [CodSpeed MCP](https://codspeed.io/docs/ai/mcp?utm_source=github&utm_medium=check&utm_content=mcp_tip_regression) with your agent.

---

<sub>Comparing <code>ct/row-fn-bool-retry</code> (2ed0398) with <code>ct/row-fn-bool-bench</code> (2449ffb)</sub>

<a href="https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?utm_source=github&utm_medium=check&utm_content=button">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://codspeed.io/pr-report/open-in-codspeed-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://codspeed.io/pr-report/open-in-codspeed-light.svg">
    <img alt="Open in CodSpeed" src="https://codspeed.io/pr-report/open-in-codspeed-light.svg" width="169" height="32">
  </picture>
</a>


[^skipped]: 293 benchmarks were skipped, so the baseline results were used instead. If they were deleted from the codebase, [click here and archive them to remove them from the performance reports](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-bool-retry?q=is%3Askipped&utm_source=github&utm_medium=check&utm_content=archive).

