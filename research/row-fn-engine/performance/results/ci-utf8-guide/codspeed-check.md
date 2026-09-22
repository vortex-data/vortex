
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
> [Open the report in CodSpeed to investigate](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?utm_source=github&utm_medium=check&utm_content=comparison_issues)

</details>

`⚡ 3` improved benchmarks  
`❌ 6` regressed benchmarks  
`✅ 2211` untouched benchmarks  
`⏩ 293` skipped benchmarks[^skipped]  


> [!WARNING]
> Please fix the performance issues or [acknowledge them on CodSpeed](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?q=is%3Aregression&utm_source=github&utm_medium=check&utm_content=acknowledge).

### Performance Changes

|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |
| --- | ---- | --------- | ------ | ------ | ---------- |
| ❌ | Simulation | [`` take_fsl_u32_random[16, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_u32_random%5B16%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 125 µs | 164.9 µs | -24.23% |
| ❌ | WallTime | [`` filtered_sink_i64_avx2[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx2%3A%3Afiltered_sink_i64_avx2%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 21.9 µs | 26.3 µs | -16.65% |
| ❌ | WallTime | [`` filtered_owned_i64_avx2[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx2%3A%3Afiltered_owned_i64_avx2%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22 µs | 26 µs | -15.63% |
| ❌ | WallTime | [`` filtered_sink_i64_avx512[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx512%3A%3Afiltered_sink_i64_avx512%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22.4 µs | 26.3 µs | -15.12% |
| ❌ | Simulation | [`` take_fsl_nullable_random[16, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_nullable_random%5B16%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 164.3 µs | 186.9 µs | -12.11% |
| ❌ | Simulation | [`` take_fsl_f16_random[256, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_f16_random%5B256%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 203.8 µs | 230.8 µs | -11.72% |
| ⚡ | Simulation | [`` take_fsl_u32_random[256, 10] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_u32_random%5B256%2C+10%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 159 µs | 123.4 µs | +28.87% |
| ⚡ | WallTime | [`` decode_avx512[8192, (Inline, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B8192%2C+%28Inline%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 93.8 µs | 83 µs | +13% |
| ⚡ | Simulation | [`` take_fsl_random[128, 10] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_random%5B128%2C+10%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 139.4 µs | 126 µs | +10.63% |

> [!TIP]
> Investigate this regression by commenting `@codspeedbot fix this regression` on this PR, or directly use the [CodSpeed MCP](https://codspeed.io/docs/ai/mcp?utm_source=github&utm_medium=check&utm_content=mcp_tip_regression) with your agent.

---

<sub>Comparing <code>ct/row-fn-utf8-decode</code> (8234800) with <code>ct/row-fn-utf8-bench</code> (50995f8)</sub>

<a href="https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?utm_source=github&utm_medium=check&utm_content=button">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://codspeed.io/pr-report/open-in-codspeed-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://codspeed.io/pr-report/open-in-codspeed-light.svg">
    <img alt="Open in CodSpeed" src="https://codspeed.io/pr-report/open-in-codspeed-light.svg" width="169" height="32">
  </picture>
</a>


[^skipped]: 293 benchmarks were skipped, so the baseline results were used instead. If they were deleted from the codebase, [click here and archive them to remove them from the performance reports](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?q=is%3Askipped&utm_source=github&utm_medium=check&utm_content=archive).

