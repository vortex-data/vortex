#!/usr/bin/env python3
# ruff: noqa: E501
import argparse
import json
import os.path
import timeit
from collections import defaultdict
from datetime import date

import polars as pl

import vortex as vx
import vortex.io

# 0: No., 1: SQL, 2: Polars
queries = [
    ("Q0", "SELECT COUNT(*) FROM hits;", lambda x: x.select(pl.len()).collect().height),
    (
        "Q1",
        "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;",
        lambda x: x.select(pl.col("AdvEngineID").filter(pl.col("AdvEngineID") != 0).count()).collect().height,
    ),
    (
        "Q2",
        "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;",
        lambda x: x.select(a_sum=pl.col("AdvEngineID").sum(), count=pl.len(), a_mean=pl.col("AdvEngineID").mean())
        .collect()
        .rows()[0],
    ),
    (
        "Q3",
        "SELECT AVG(UserID) FROM hits;",
        lambda x: x.select(pl.col("UserID").mean()).collect().item(),
    ),
    (
        "Q4",
        "SELECT COUNT(DISTINCT UserID) FROM hits;",
        lambda x: x.select(pl.col("UserID").n_unique()).collect().item(),
    ),
    (
        "Q5",
        "SELECT COUNT(DISTINCT SearchPhrase) FROM hits;",
        lambda x: x.select(pl.col("SearchPhrase").n_unique()).collect().item(),
    ),
    (
        "Q6",
        "SELECT MIN(EventDate), MAX(EventDate) FROM hits;",
        lambda x: x.select(e_min=pl.col("EventDate").min(), e_max=pl.col("EventDate").max()).collect().rows()[0],
    ),
    (
        "Q7",
        "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;",
        lambda x: x.filter(pl.col("AdvEngineID") != 0)
        .group_by("AdvEngineID")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(),
    ),
    (
        "Q8",
        "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;",
        lambda x: x.group_by("RegionID")
        .agg(pl.col("UserID").n_unique())
        .sort("UserID", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q9",
        "SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;",
        lambda x: x.group_by("RegionID")
        .agg(
            [
                pl.sum("AdvEngineID").alias("AdvEngineID_sum"),
                pl.mean("ResolutionWidth").alias("ResolutionWidth_mean"),
                pl.col("UserID").n_unique().alias("UserID_nunique"),
            ]
        )
        .sort("AdvEngineID_sum", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q10",
        "SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;",
        lambda x: x.filter(pl.col("MobilePhoneModel") != "")
        .group_by("MobilePhoneModel")
        .agg(pl.col("UserID").n_unique())
        .sort("UserID", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q11",
        "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;",
        lambda x: x.filter(pl.col("MobilePhoneModel") != "")
        .group_by(["MobilePhone", "MobilePhoneModel"])
        .agg(pl.col("UserID").n_unique())
        .sort("UserID", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q12",
        "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .group_by("SearchPhrase")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q13",
        "SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .group_by("SearchPhrase")
        .agg(pl.col("UserID").n_unique())
        .sort("UserID", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q14",
        "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .group_by(["SearchEngineID", "SearchPhrase"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q15",
        "SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;",
        lambda x: x.group_by("UserID").agg(pl.len().alias("count")).sort("count", descending=True).head(10).collect(),
    ),
    (
        "Q16",
        "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;",
        lambda x: x.group_by(["UserID", "SearchPhrase"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q17",
        "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;",
        lambda x: x.group_by(["UserID", "SearchPhrase"]).agg(pl.len()).head(10).collect(),
    ),
    (
        "Q18",
        "SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;",
        lambda x: x.group_by([pl.col("UserID"), pl.col("EventTime").dt.minute(), "SearchPhrase"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q19",
        "SELECT UserID FROM hits WHERE UserID = 435090932899640449;",
        lambda x: x.select("UserID").filter(pl.col("UserID") == 435090932899640449).collect(),
    ),
    (
        "Q20",
        "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';",
        lambda x: x.filter(pl.col("URL").str.contains("google")).select(pl.len()).collect().item(),
    ),
    (
        "Q21",
        "SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter((pl.col("URL").str.contains("google")) & (pl.col("SearchPhrase") != ""))
        .group_by("SearchPhrase")
        .agg([pl.col("URL").min(), pl.len().alias("count")])
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q22",
        "SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter(
            (pl.col("Title").str.contains("Google"))
            & (~pl.col("URL").str.contains(".google."))
            & (pl.col("SearchPhrase") != "")
        )
        .group_by("SearchPhrase")
        .agg(
            [
                pl.col("URL").min(),
                pl.col("Title").min(),
                pl.len().alias("count"),
                pl.col("UserID").n_unique(),
            ]
        )
        .sort("count", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q23",
        "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;",
        lambda x: x.filter(pl.col("URL").str.contains("google")).sort("EventTime").head(10).collect(),
    ),
    (
        "Q24",
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "").sort("EventTime").select("SearchPhrase").head(10).collect(),
    ),
    (
        "Q25",
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "").sort("SearchPhrase").select("SearchPhrase").head(10).collect(),
    ),
    (
        "Q26",
        "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .sort(["EventTime", "SearchPhrase"])
        .select("SearchPhrase")
        .head(10)
        .collect(),
    ),
    (
        "Q27",
        "SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;",
        lambda x: x.filter(pl.col("URL") != "")  # WHERE URL <> ''
        .group_by("CounterID")  # GROUP BY CounterID
        .agg(
            [
                pl.col("URL").str.len_chars().mean().alias("l"),  # AVG(STRLEN(URL))
                pl.len().alias("c"),  # COUNT(*)
            ]
        )
        .filter(pl.col("c") > 100000)  # HAVING COUNT(*) > 100000
        .sort("l", descending=True)  # ORDER BY l DESC
        .limit(25)
        .collect(),  # LIMIT 25,
    ),
    (
        "Q28",
        "SELECT REGEXP_REPLACE(Referer, '(?-u)^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(STRLEN(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;",
        lambda x: (
            x.filter(pl.col("Referer") != "")
            .with_columns(pl.col("Referer").str.extract(r"^https?://(?:www\\.)?([^/]+)/.*$").alias("k"))
            .group_by("k")
            .agg(
                [
                    pl.col("Referer").str.len_chars().mean().alias("l"),  # AVG(STRLEN(Referer))
                    pl.col("Referer").min().alias("min_referer"),  # MIN(Referer)
                    pl.len().alias("c"),  # COUNT(*)
                ]
            )
            .filter(pl.col("c") > 100000)  # HAVING COUNT(*) > 100000
            .sort("l", descending=True)  # ORDER BY l DESC
            .limit(25)
            .collect()  # LIMIT 25
        ),
    ),
    (
        "Q29",
        "SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 2), SUM(ResolutionWidth + 3), SUM(ResolutionWidth + 4), SUM(ResolutionWidth + 5), SUM(ResolutionWidth + 6), SUM(ResolutionWidth + 7), SUM(ResolutionWidth + 8), SUM(ResolutionWidth + 9), SUM(ResolutionWidth + 10), SUM(ResolutionWidth + 11), SUM(ResolutionWidth + 12), SUM(ResolutionWidth + 13), SUM(ResolutionWidth + 14), SUM(ResolutionWidth + 15), SUM(ResolutionWidth + 16), SUM(ResolutionWidth + 17), SUM(ResolutionWidth + 18), SUM(ResolutionWidth + 19), SUM(ResolutionWidth + 20), SUM(ResolutionWidth + 21), SUM(ResolutionWidth + 22), SUM(ResolutionWidth + 23), SUM(ResolutionWidth + 24), SUM(ResolutionWidth + 25), SUM(ResolutionWidth + 26), SUM(ResolutionWidth + 27), SUM(ResolutionWidth + 28), SUM(ResolutionWidth + 29), SUM(ResolutionWidth + 30), SUM(ResolutionWidth + 31), SUM(ResolutionWidth + 32), SUM(ResolutionWidth + 33), SUM(ResolutionWidth + 34), SUM(ResolutionWidth + 35), SUM(ResolutionWidth + 36), SUM(ResolutionWidth + 37), SUM(ResolutionWidth + 38), SUM(ResolutionWidth + 39), SUM(ResolutionWidth + 40), SUM(ResolutionWidth + 41), SUM(ResolutionWidth + 42), SUM(ResolutionWidth + 43), SUM(ResolutionWidth + 44), SUM(ResolutionWidth + 45), SUM(ResolutionWidth + 46), SUM(ResolutionWidth + 47), SUM(ResolutionWidth + 48), SUM(ResolutionWidth + 49), SUM(ResolutionWidth + 50), SUM(ResolutionWidth + 51), SUM(ResolutionWidth + 52), SUM(ResolutionWidth + 53), SUM(ResolutionWidth + 54), SUM(ResolutionWidth + 55), SUM(ResolutionWidth + 56), SUM(ResolutionWidth + 57), SUM(ResolutionWidth + 58), SUM(ResolutionWidth + 59), SUM(ResolutionWidth + 60), SUM(ResolutionWidth + 61), SUM(ResolutionWidth + 62), SUM(ResolutionWidth + 63), SUM(ResolutionWidth + 64), SUM(ResolutionWidth + 65), SUM(ResolutionWidth + 66), SUM(ResolutionWidth + 67), SUM(ResolutionWidth + 68), SUM(ResolutionWidth + 69), SUM(ResolutionWidth + 70), SUM(ResolutionWidth + 71), SUM(ResolutionWidth + 72), SUM(ResolutionWidth + 73), SUM(ResolutionWidth + 74), SUM(ResolutionWidth + 75), SUM(ResolutionWidth + 76), SUM(ResolutionWidth + 77), SUM(ResolutionWidth + 78), SUM(ResolutionWidth + 79), SUM(ResolutionWidth + 80), SUM(ResolutionWidth + 81), SUM(ResolutionWidth + 82), SUM(ResolutionWidth + 83), SUM(ResolutionWidth + 84), SUM(ResolutionWidth + 85), SUM(ResolutionWidth + 86), SUM(ResolutionWidth + 87), SUM(ResolutionWidth + 88), SUM(ResolutionWidth + 89) FROM hits;",
        lambda x: x.select(pl.sum_horizontal([pl.col("ResolutionWidth").shift(i) for i in range(1, 90)])).collect(),
    ),
    (
        "Q30",
        "SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .group_by(["SearchEngineID", "ClientIP"])
        .agg(
            [
                pl.len().alias("c"),
                pl.sum("IsRefresh").alias("IsRefreshSum"),
                pl.mean("ResolutionWidth").alias("AvgResolutionWidth"),
            ]
        )
        .sort("c", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q31",
        "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;",
        lambda x: x.filter(pl.col("SearchPhrase") != "")
        .group_by(["WatchID", "ClientIP"])
        .agg(
            [
                pl.len().alias("c"),
                pl.sum("IsRefresh").alias("IsRefreshSum"),
                pl.mean("ResolutionWidth").alias("AvgResolutionWidth"),
            ]
        )
        .sort("c", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q32",
        "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;",
        lambda x: x.group_by(["WatchID", "ClientIP"])
        .agg(
            [
                pl.len().alias("c"),
                pl.sum("IsRefresh").alias("IsRefreshSum"),
                pl.mean("ResolutionWidth").alias("AvgResolutionWidth"),
            ]
        )
        .sort("c", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q33",
        "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;",
        lambda x: x.group_by("URL").agg(pl.len().alias("c")).sort("c", descending=True).head(10).collect(),
    ),
    (
        "Q34",
        "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;",
        lambda x: x.group_by("URL").agg(pl.len().alias("c")).sort("c", descending=True).head(10).collect(),
    ),
    (
        "Q35",
        "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;",
        lambda x: x.with_columns([pl.col("ClientIP")])
        .group_by(["ClientIP"])
        .agg(pl.len().alias("c"))
        .sort("c", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q36",
        "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("DontCountHits") == 0)
            & (pl.col("IsRefresh") == 0)
            & (pl.col("URL") != "")
        )
        .group_by("URL")
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q37",
        "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("DontCountHits") == 0)
            & (pl.col("IsRefresh") == 0)
            & (pl.col("Title") != "")
        )
        .group_by("Title")
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .head(10)
        .collect(),
    ),
    (
        "Q38",
        "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("IsRefresh") == 0)
            & (pl.col("IsLink") != 0)
            & (pl.col("IsDownload") == 0)
        )
        .group_by("URL")
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .slice(1000, 10)
        .collect(),
    ),
    (
        "Q39",
        "SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("IsRefresh") == 0)
        )
        .group_by(
            [
                "TraficSourceID",
                "SearchEngineID",
                "AdvEngineID",
                pl.when(pl.col("SearchEngineID").eq(0) & pl.col("AdvEngineID").eq(0))
                .then(pl.col("Referer"))
                .otherwise(pl.lit(""))
                .alias("Src"),
                "URL",
            ]
        )
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .slice(1000, 10)
        .collect(),
    ),
    (
        "Q40",
        "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("IsRefresh") == 0)
            & (pl.col("TraficSourceID").is_in([-1, 6]))
            & (pl.col("RefererHash") == 3594120000172545465)
        )
        .group_by(["URLHash", "EventDate"])
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .slice(100, 10)
        .collect(),
    ),
    (
        "Q41",
        "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 1))
            & (pl.col("EventDate") <= date(2013, 7, 31))
            & (pl.col("IsRefresh") == 0)
            & (pl.col("DontCountHits") == 0)
            & (pl.col("URLHash") == 2868770270353813622)
        )
        .group_by(["WindowClientWidth", "WindowClientHeight"])
        .agg(pl.len().alias("PageViews"))
        .sort("PageViews", descending=True)
        .slice(10000, 10)
        .collect(),
    ),
    (
        "Q42",
        "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;",
        lambda x: x.filter(
            (pl.col("CounterID") == 62)
            & (pl.col("EventDate") >= date(2013, 7, 14))
            & (pl.col("EventDate") <= date(2013, 7, 15))
            & (pl.col("IsRefresh") == 0)
            & (pl.col("DontCountHits") == 0)
        )
        .group_by(pl.col("EventTime").dt.truncate("1m"))
        .agg(pl.len().alias("PageViews"))
        .slice(1000, 10)
        .collect(),
    ),
]


def run_timings(lf: pl.LazyFrame, name: str, src: str, load_time: int | None) -> None:
    queries_times = []
    for q in queries:
        # if q[0] == "Q19":
        #    print("Q19 fails for https://github.com/pola-rs/polars/issues/21674")
        #    queries_times.append([None, None, None])
        #    continue
        print(q[0])
        times = []
        for _ in range(3):
            start = timeit.default_timer()
            try:
                result = q[2](lf)
            except Exception as e:
                print("Failed", e)
                result = None

            end = timeit.default_timer()
            if result is None:
                times.append(None)
            else:
                times.append(end - start)
        queries_times.append(times)

    result_json = {
        "system": name,
        "date": date.today().strftime("%Y-%m-%d"),
        "machine": "c6a.metal, 500gb gp2",
        "cluster_size": 1,
        "comment": "",
        "tags": [
            "column-oriented",
            src,
        ],
        "load_time": float(load_time) if load_time is not None else None,
        "result": queries_times,
    }
    print(result_json)
    return result_json

    # if cpuinfo contains "AMD EPYC 9654" update machine and write result into results/epyc-9654.json
    if "AMD EPYC 9654" in open("/proc/cpuinfo").read():
        result_json["machine"] = "EPYC 9654, 384G"
        with open(f"results/{src}_epyc-9654.json", "w") as f:
            f.write(json.dumps(result_json, indent=4))
    else:
        # write result into results/c6a.metal.json
        with open(f"results/{src}_c6a.metal.json", "w") as f:
            f.write(json.dumps(result_json, indent=4))


PARSER = argparse.ArgumentParser()

PARSER.add_argument(
    "--path",
    type=str,
    default="hits.parquet",
    help="Path to the parquet file",
)

PARSER.add_argument(
    "--formats",
    nargs="+",
    choices=("vortex", "parquet"),
    default=None,
    help="Formats to run",
)

PARSER.add_argument(
    "-q",
    "--queries",
    nargs="+",
    type=int,
    default=list(range(len(queries))),
    help="Queries to run",
)

PARSER.add_argument("-i", "--iterations", type=int, default=3, help="Number of iterations to run")


def main(args):
    if not os.path.exists(args.path):
        raise ValueError(f"File {args.path} does not exist")

    results = defaultdict(list)

    def run_queries(format: str, lf: pl.LazyFrame):
        for q in args.queries:
            timings = []
            for _ in range(args.iterations):
                start = timeit.default_timer()
                try:
                    _result = queries[q][2](lf)
                except Exception as e:
                    print(f"Failed Q{q}", e)
                timings.append(timeit.default_timer() - start)
            average = sum(timings) / len(timings)
            results[format].append(average)
            print(f"{format} Q{q}", average)

    if args.formats is None or "vortex" in args.formats:
        vx_base, _ = os.path.splitext(args.path)
        vx_path = f"{vx_base}.vortex"

        if not os.path.exists(vx_path):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(args.path)

            def _iter():
                for i in range(pf.num_row_groups):
                    arr = pf.read_row_group(i).to_struct_array()
                    arr = vx.Array.from_arrow(arr)
                    yield arr

            it = vx.ArrayIterator.from_iter(vx.DType.from_arrow(pf.schema_arrow, non_nullable=True), _iter())
            vx.io.write(it, vx_path)

        lf = vx.open(vx_path).to_polars()
        run_queries("vortex", lf)

    if args.formats is None or "parquet" in args.formats:
        lf = pl.scan_parquet(args.path)
        run_queries("parquet", lf)

        for format in list(results):
            if format == "parquet":
                continue
            results[f"{format}_vs_pq"] = [a / b for a, b in zip(results[format], results["parquet"])]

    print(results)


if __name__ == "__main__":
    main(PARSER.parse_args())
