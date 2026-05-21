#!/usr/bin/env python3
"""Canonical B200/GH200 OnPair-vs-nvCOMP results store (CSV + JSON) + rendered
markdown tables in B200_PRELIMINARY.md.

PRELIMINARY — unlocked clocks, single runs, NCU unavailable (see banner).

OnPair B200 rows are read live from `vortex-bench/data/onpair-bench/summary.json`
(self-validating). nvCOMP HW rows come from the standalone `nvcomp_hw_bench.cu`
(chunk 256 KiB). Each HW codec has two presets:
  - hi   = max compression ratio  (Deflate algorithm=5)
  - fast = best (de)compression throughput (Deflate algorithm=0, "entropy-only,
           symmetric"); LZ4 has no level so it is a single throughput codec.
Run: `python benchmarks/onpair-bench/gen_b200_tables.py`
"""
import csv, json, pathlib

GIB2GB = 1.073741824
HERE = pathlib.Path(__file__).resolve().parent
SUMMARY = HERE.parent.parent / "vortex-bench/data/onpair-bench/summary.json"

FIELDS = ["platform","dataset","column","bits","decoder","preset","backend",
          "ratio","compress_gib_s","decode_gib_s","kernel","note"]

BIG = [("clickbench","URL"),("fineweb","text"),("wikipedia","text"),
       ("tpch-sf10","l_comment"),("tpch-sf10","ps_comment")]

def onpair_b200_rows():
    rows=[]
    for e in json.load(open(SUMMARY)):
        g=e["gpu"]
        rows.append(dict(platform="B200",dataset=e["dataset_id"],column=e["column"],
            bits=e["bits"],decoder="OnPair",preset="-",backend="cuda",
            ratio=round(e["mem_ratio"],2),compress_gib_s=round(e["encode_gib_s"],3),
            decode_gib_s=round(g["best_decode_gib_s"],1),
            kernel=g["best_kernel"].replace("onpair_shmem_",""),note=""))
    return rows

# nvCOMP HW (chunk 256 KiB). value = (ratio, compress_gib_s, decode_gib_s)
NVCOMP = {
  ("tpch-sf10","l_comment"): {"Deflate-hi":(4.56,0.4,293.0),"Deflate-fast":(1.85,47.7,121.7),"LZ4":(2.17,13.0,224.1)},
  ("fineweb","text"):        {"Deflate-hi":(2.55,0.5,169.8),"Deflate-fast":(1.71,64.4,125.6),"LZ4":(1.54,10.9,188.3)},
  ("wikipedia","text"):      {"Deflate-hi":(2.70,0.5,175.6),"Deflate-fast":(1.67,80.6,123.8),"LZ4":(1.64, 8.7,194.3)},
  ("clickbench","URL"):      {"Deflate-hi":(6.44,0.4,383.0),"Deflate-fast":(1.45,62.4,125.7),"LZ4":(3.70,23.5,362.7)},
  ("tpch-sf10","ps_comment"):{"Deflate-hi":(5.67,0.5,377.6),"Deflate-fast":(1.85,63.7,124.7),"LZ4":(2.56,15.5,246.6)},
}
# Zstd CUDA backend (no HW path). compress is CPU-side (not measured). decode from
# onpair-chunk-bench level sweep; frame-size artifact on long-string cols (flagged).
ZSTD = {
  ("tpch-sf10","l_comment"): {"hi(L3)":(2.87,84.5,""),"fast(L-10)":(1.79,94.7,"")},
  ("clickbench","URL"):      {"hi(L3)":(5.64,112.1,"")},
  ("tpch-sf10","ps_comment"):{"hi(L3)":(4.16,37.4,"")},
  ("fineweb","text"):        {"hi(L3)":(2.57,8.3,"frame-size artifact (long strings)")},
  ("wikipedia","text"):      {"hi(L3)":(2.74,1.0,"frame-size artifact (long strings)")},
}

def nvcomp_rows():
    rows=[]
    for (ds,col),presets in NVCOMP.items():
        for name,(r,c,d) in presets.items():
            codec="nvCOMP-"+name.split("-")[0]
            preset = name.split("-")[1] if "-" in name else "fast"
            rows.append(dict(platform="B200",dataset=ds,column=col,bits=None,
                decoder=codec,preset=preset,backend="hardware",ratio=r,
                compress_gib_s=c,decode_gib_s=d,kernel="DE",
                note="deflate algo5,256KiB" if preset=="hi" and codec=="nvCOMP-Deflate"
                     else ("deflate algo0,256KiB" if preset=="fast" else "256KiB")))
    for (ds,col),presets in ZSTD.items():
        for name,(r,d,nt) in presets.items():
            rows.append(dict(platform="B200",dataset=ds,column=col,bits=None,
                decoder="nvCOMP-Zstd",preset=name,backend="cuda",ratio=r,
                compress_gib_s=None,decode_gib_s=d,kernel="cuda",
                note=(nt+"; " if nt else "")+"CUDA backend (no HW path), CPU-side compress"))
    # Zstd HARDWARE unsupported sentinel
    rows.append(dict(platform="B200",dataset="*",column="*",bits=None,decoder="nvCOMP-Zstd",
        preset="hardware",backend="hardware",ratio=None,compress_gib_s=None,decode_gib_s=None,
        kernel="DE",note="UNSUPPORTED (DE has no Zstd; status 10)"))
    return rows

# OnPair GH200 (handover Section 5; auto kernel). (ratio, decode)
GH200 = {
  ("fineweb","text",12):(2.24,567.0,"4tpt_split8read"),("fineweb","text",16):(2.89,470.0,"4tpt"),
  ("wikipedia","text",12):(2.15,538.0,"4tpt_split8read"),("wikipedia","text",16):(2.80,538.0,"4tpt"),
  ("tpch-sf10","ps_comment",12):(6.23,1117.0,"4tpt"),("tpch-sf10","ps_comment",16):(5.82,866.0,"4tpt"),
  ("book-reviews","text",12):(None,607.0,"4tpt_split8read"),
}
def gh200_rows():
    return [dict(platform="GH200",dataset=ds,column=col,bits=b,decoder="OnPair",preset="-",
        backend="cuda",ratio=r,compress_gib_s=None,decode_gib_s=d,kernel=k,
        note="carried from handover §5") for (ds,col,b),(r,d,k) in GH200.items()]

def fmt(v,nd=1):
    if v is None or v=="": return "—"
    return f"{v:.{nd}f}" if isinstance(v,float) else str(v)

def main():
    data = onpair_b200_rows()+nvcomp_rows()+gh200_rows()
    for d in data:
        d["decode_gb_s"]=round(d["decode_gib_s"]*GIB2GB,1) if isinstance(d["decode_gib_s"],(int,float)) else None
    with open(HERE/"b200_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS+["decode_gb_s"]); w.writeheader()
        for d in data: w.writerow(d)
    json.dump(data,open(HERE/"b200_results.json","w"),indent=2)

    L=[]
    L.append("# ⚠️ PRELIMINARY — OnPair vs nvCOMP, GH200 vs B200\n")
    L.append("> **PRELIMINARY (2026-05-21).** Unlocked GPU clocks (±~1–5% on absolutes), single "
             "runs, NCU unavailable in-container (`ERR_NVGPUCTRPERM`). Re-measure with locked clocks "
             "before quoting.\n")
    L.append("> Source of truth: `b200_results.csv` / `.json` (regen `gen_b200_tables.py`). OnPair "
             "rows read live from `summary.json`; nvCOMP HW from `nvcomp_hw_bench.cu` (chunk 256 KiB).\n")
    L.append("> Decode/compress = GiB/s over uncompressed bytes, 100 iters. nvCOMP HW = Blackwell "
             "hardware Decompression Engine, byte-exact. Compression ratio is hardware-independent.\n")
    L.append("> **Each nvCOMP HW codec has two presets:** `hi` = max ratio (Deflate algo5); "
             "`fast` = best (de)compression throughput (Deflate algo0; LZ4 is single-pass, no level).\n")

    # 1. OnPair all columns
    L.append("\n## 1. OnPair — B200, all columns (best kernel)\n")
    L.append("| dataset/column | bits | ratio | compress GiB/s | decode GiB/s | decode GB/s | kernel |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for d in sorted([x for x in data if x["platform"]=="B200" and x["decoder"]=="OnPair"],
                    key=lambda x:(x["dataset"],x["column"],x["bits"])):
        L.append(f"| {d['dataset']}/{d['column']} | {d['bits']} | {fmt(d['ratio'],2)}× | "
                 f"{fmt(d['compress_gib_s'],3)} | {fmt(d['decode_gib_s'])} | {fmt(d['decode_gb_s'])} | {d['kernel']} |")

    # 2. nvCOMP HW presets (the new ask)
    L.append("\n## 2. nvCOMP hardware-engine presets — big columns (ratio · compress GiB/s · decode GiB/s)\n")
    L.append("| dataset/column | Deflate-hi (max ratio) | Deflate-fast (max throughput) | LZ4 (single) |")
    L.append("| --- | --- | --- | --- |")
    def nv(ds,col,dec,pre):
        for d in data:
            if d["platform"]=="B200" and d["dataset"]==ds and d["column"]==col and d["decoder"]==dec and d["preset"]==pre:
                return f"{fmt(d['ratio'],2)}× · {fmt(d['compress_gib_s'])} · {fmt(d['decode_gib_s'],0)}"
        return "—"
    for ds,col in BIG:
        L.append(f"| {ds}/{col} | {nv(ds,col,'nvCOMP-Deflate','hi')} | "
                 f"{nv(ds,col,'nvCOMP-Deflate','fast')} | {nv(ds,col,'nvCOMP-LZ4','fast')} |")
    L.append("\n*Format: `ratio× · compress GiB/s · decode GiB/s`. Deflate-hi gives best ratio + fast "
             "decode but ~0.5 GiB/s compress; Deflate-fast compresses ~50–80 GiB/s but low ratio/decode; "
             "LZ4 is the balanced middle.*\n")

    # 2b. Zstd CUDA presets
    L.append("\n## 2b. nvCOMP Zstd (CUDA backend — no HW path) presets (ratio · decode GiB/s)\n")
    L.append("| dataset/column | hi (level 3) | fast (level −10) | note |")
    L.append("| --- | --- | --- | --- |")
    for ds,col in BIG:
        hi=fast=note="—"
        for d in data:
            if d["platform"]=="B200" and d["dataset"]==ds and d["column"]==col and d["decoder"]=="nvCOMP-Zstd":
                if d["preset"].startswith("hi"): hi=f"{fmt(d['ratio'],2)}× · {fmt(d['decode_gib_s'],0)}";
                if d["preset"].startswith("fast"): fast=f"{fmt(d['ratio'],2)}× · {fmt(d['decode_gib_s'],0)}"
                if "artifact" in d["note"]: note="frame-size artifact (long strings)"
        L.append(f"| {ds}/{col} | {hi} | {fast} | {note} |")
    L.append("\n*Zstd has no hardware-engine path (DE returns status 10). CUDA-backend decode is "
             "frame-size-sensitive: long-string columns (fineweb/wikipedia) collapse to <10 GiB/s "
             "because fixed values-per-frame makes huge frames. Compress is CPU-side (not comparable).*\n")

    # 3. headline comparison
    L.append("\n## 3. Headline — OnPair vs best nvCOMP HW (big columns)\n")
    L.append("| dataset/column | OnPair best (ratio · decode) | Deflate-hi (ratio · decode) | OnPair decode advantage |")
    L.append("| --- | --- | --- | ---: |")
    for ds,col in BIG:
        ops=[d for d in data if d["platform"]=="B200" and d["dataset"]==ds and d["column"]==col and d["decoder"]=="OnPair"]
        best=max(ops,key=lambda x:x["decode_gib_s"])
        dh=next((d for d in data if d["platform"]=="B200" and d["dataset"]==ds and d["column"]==col and d["decoder"]=="nvCOMP-Deflate" and d["preset"]=="hi"),None)
        adv=f"{best['decode_gib_s']/dh['decode_gib_s']:.1f}×" if dh else "—"
        L.append(f"| {ds}/{col} | {fmt(best['ratio'],2)}× · {fmt(best['decode_gib_s'],0)} (b{best['bits']}) | "
                 f"{fmt(dh['ratio'],2)}× · {fmt(dh['decode_gib_s'],0)} | {adv} |")
    L.append("\n*Deflate-hi beats OnPair on ratio for l_comment (4.56 vs 4.17) and clickbench URL "
             "(6.44 vs 3.86); OnPair wins ratio elsewhere and wins decode throughput everywhere.*\n")

    # 4. GH200 vs B200
    L.append("\n## 4. OnPair GH200 vs B200 (decode GiB/s)\n")
    L.append("| dataset/column | bits | ratio | GH200 | B200 | Δ |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    b200={(d["dataset"],d["column"],d["bits"]):d for d in data if d["platform"]=="B200" and d["decoder"]=="OnPair"}
    for (ds,col,b),(r,gd,gk) in sorted(GH200.items()):
        bd=b200.get((ds,col,b))
        if bd:
            L.append(f"| {ds}/{col} | {b} | {fmt(bd['ratio'],2)}× | {fmt(gd,0)} ({gk}) | {fmt(bd['decode_gib_s'],0)} | "
                     f"{(bd['decode_gib_s']/gd-1)*100:+.0f}% |")
        else:
            L.append(f"| {ds}/{col} | {b} | — | {fmt(gd,0)} ({gk}) | not on B200 | — |")

    # 5. validation + gaps
    L.append("\n## 5. Validation & known gaps\n")
    L.append("- **OnPair, 22/22**: read directly from `summary.json` (ratio=mem_ratio, decode=best, kernel=best) — no transcription.\n")
    L.append("- **nvCOMP HW**: all decodes byte-exact (`valid=YES`); ratios reproduce exactly, throughput ±~4% (clock noise).\n")
    L.append("- **GH200, 7/7**: match handover §5 exactly. book-reviews has no public source → not re-run on B200.\n")
    L.append("- **Deflate level matters**: SDK default algo=1 (\"low ratio\") understated ratio AND decode; presets above use algo5/algo0. Chunk size is not a ratio lever (Deflate window caps at 32 KiB).\n")
    L.append("- **Zstd CUDA** frame-size artifact on long-string columns; **Zstd HW unsupported** (status 10).\n")
    L.append("- **Caveat**: OnPair ratio = whole-column dictionary over 1000 MB chunks; nvCOMP ratio = batched 256 KiB chunks. Both realistic, different granularity.\n")
    L.append("- NCU mechanistic limiter analysis blocked (no `CAP_SYS_ADMIN`).\n")

    (HERE/"B200_PRELIMINARY.md").write_text("\n".join(L)+"\n")
    print(f"wrote csv/json/md — {len(data)} records, {sum(1 for d in data if d['decoder']=='OnPair' and d['platform']=='B200')} OnPair B200 cells")

if __name__=="__main__":
    main()
