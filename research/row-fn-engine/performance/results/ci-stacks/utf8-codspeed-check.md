
<details>
<summary>:warning: <b>Unknown Walltime execution environment detected</b></summary>

> Using the Walltime instrument on standard Hosted Runners will lead to inconsistent data.
>
> For the most accurate results, we recommend using [CodSpeed Macro Runners](https://codspeed.io/docs/instruments/walltime): bare-metal machines fine-tuned for performance measurement consistency.

</details>

`⚡ 27` improved benchmarks  
`❌ 7` regressed benchmarks  
`✅ 2210` untouched benchmarks  
`⏩ 293` skipped benchmarks[^skipped]  


> [!WARNING]
> Please fix the performance issues or [acknowledge them on CodSpeed](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?q=is%3Aregression&utm_source=github&utm_medium=check&utm_content=acknowledge).

### Performance Changes

|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |
| --- | ---- | --------- | ------ | ------ | ---------- |
| ❌ | Simulation | [`` take_fsl_f16_random[16, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_f16_random%5B16%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 118.4 µs | 176 µs | -32.71% |
| ❌ | WallTime | [`` filtered_sink_i64_avx2[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx2%3A%3Afiltered_sink_i64_avx2%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 21.9 µs | 26.4 µs | -16.89% |
| ❌ | WallTime | [`` filtered_sink_i64_avx512[OneNullInEight] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_output.rs%3A%3Aavx512%3A%3Afiltered_sink_i64_avx512%5BOneNullInEight%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 22.3 µs | 26.4 µs | -15.39% |
| ❌ | WallTime | [`` decode_avx512[16384, (External, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B16384%2C+%28External%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 288.5 µs | 338 µs | -14.63% |
| ❌ | Simulation | [`` take_fsl_u8_random[256, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_u8_random%5B256%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 156.3 µs | 181.7 µs | -13.95% |
| ❌ | WallTime | [`` decode_avx512[16384, (External, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B16384%2C+%28External%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 272.3 µs | 314.8 µs | -13.49% |
| ❌ | Simulation | [`` take_fsl_nullable_random[16, 100] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Ftake_fsl.rs%3A%3Atake_fsl_nullable_random%5B16%2C+100%5D&runnerMode=Simulation&utm_source=github&utm_medium=check&utm_content=benchmark) | 164.3 µs | 186.9 µs | -12.1% |
| ⚡ | WallTime | [`` decode_neon[1, (Inline, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aneon%3A%3Adecode_neon%5B1%2C+%28Inline%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 566 ns | 120 ns | ×4.7 |
| ⚡ | WallTime | [`` decode_avx512[1, (Inline, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B1%2C+%28Inline%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 556 ns | 125 ns | ×4.4 |
| ⚡ | WallTime | [`` decode_avx2[1, (Inline, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx2%3A%3Adecode_avx2%5B1%2C+%28Inline%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 509 ns | 126 ns | ×4 |
| ⚡ | WallTime | [`` decode_avx512[1, (External, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B1%2C+%28External%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 598 ns | 157 ns | ×3.8 |
| ⚡ | WallTime | [`` decode_neon[1, (Inline, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aneon%3A%3Adecode_neon%5B1%2C+%28Inline%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 394 ns | 111 ns | ×3.5 |
| ⚡ | WallTime | [`` decode_avx2[1, (Inline, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx2%3A%3Adecode_avx2%5B1%2C+%28Inline%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 357 ns | 105 ns | ×3.4 |
| ⚡ | WallTime | [`` decode_avx2[1, (External, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx2%3A%3Adecode_avx2%5B1%2C+%28External%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 507 ns | 164 ns | ×3.1 |
| ⚡ | WallTime | [`` decode_neon[1, (External, OneNullInEight)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aneon%3A%3Adecode_neon%5B1%2C+%28External%2C+OneNullInEight%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 599 ns | 201 ns | ×3 |
| ⚡ | WallTime | [`` decode_avx512[1, (Inline, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B1%2C+%28Inline%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 375 ns | 127 ns | ×3 |
| ⚡ | WallTime | [`` decode_avx512[1, (External, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx512%3A%3Adecode_avx512%5B1%2C+%28External%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 509 ns | 192 ns | ×2.7 |
| ⚡ | WallTime | [`` decode_avx2[1, (External, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aavx2%3A%3Adecode_avx2%5B1%2C+%28External%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 495 ns | 189 ns | ×2.6 |
| ⚡ | WallTime | [`` decode_neon[1, (External, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aneon%3A%3Adecode_neon%5B1%2C+%28External%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 500 ns | 212 ns | ×2.4 |
| ⚡ | WallTime | [`` decode_neon[64, (Inline, AllValid)] ``](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?uri=vortex-array%2Fbenches%2Frow_fn_utf8.rs%3A%3Aneon%3A%3Adecode_neon%5B64%2C+%28Inline%2C+AllValid%29%5D&runnerMode=WallTime&utm_source=github&utm_medium=check&utm_content=benchmark) | 808 ns | 543 ns | +48.8% |
| ... | ... | ... | ... | ... | ... |

<br/>

> :information_source: _Only the first 20 benchmarks are displayed. [Go to the app to view all benchmarks](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?utm_source=github&utm_medium=check&utm_content=all_benchmarks)._

> [!TIP]
> Investigate this regression by commenting `@codspeedbot fix this regression` on this PR, or directly use the [CodSpeed MCP](https://codspeed.io/docs/ai/mcp?utm_source=github&utm_medium=check&utm_content=mcp_tip_regression) with your agent.

---

<sub>Comparing <code>ct/row-fn-utf8-decode</code> (97f8c26) with <code>ct/row-fn-utf8-bench</code> (6c33a30)</sub>

<a href="https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?utm_source=github&utm_medium=check&utm_content=button">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://codspeed.io/pr-report/open-in-codspeed-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://codspeed.io/pr-report/open-in-codspeed-light.svg">
    <img alt="Open in CodSpeed" src="https://codspeed.io/pr-report/open-in-codspeed-light.svg" width="169" height="32">
  </picture>
</a>


[^skipped]: 293 benchmarks were skipped, so the baseline results were used instead. If they were deleted from the codebase, [click here and archive them to remove them from the performance reports](https://app.codspeed.io/vortex-data/vortex/branches/ct%2Frow-fn-utf8-decode?q=is%3Askipped&utm_source=github&utm_medium=check&utm_content=archive).

