-- Some basic string-focused queries against the HuggingFace FineWeb dataset, numbered from
-- Q0 in file order. The harness splits the file on semicolons, so a comment must never
-- contain one.

-- Q0: simple summary.
SELECT count(DISTINCT dump) FROM fineweb;

-- Q1: selective string equality filter.
SELECT * FROM fineweb WHERE dump = 'CC-MAIN-2016-30';

-- Q2: LIKE with prefix filter.
SELECT * FROM fineweb WHERE date LIKE '2020-10-%';

-- Q3: LIKE with simple containment filter.
SELECT * FROM fineweb WHERE url LIKE '%google%' AND text LIKE '%Google%';

-- Q4: LIKE with larger containment filter.
SELECT * FROM fineweb WHERE url LIKE '%.google.%' OR text LIKE '% Google %';

-- Q5: LIKE with rare containment filter.
SELECT * FROM fineweb WHERE text LIKE '% vortex %';

-- Q6: more LIKE filters.
SELECT * FROM fineweb WHERE url LIKE '%espn%' AND language = 'en' AND language_score > 0.92;

-- Q7: more LIKE filters.
SELECT * FROM fineweb WHERE url LIKE '%espn%' OR url LIKE '%www.espn.go.com%' OR url LIKE '%espn.go.com%';

-- Q8: no results, stats cannot prune but tokenized bloom filters could.
SELECT * FROM fineweb WHERE file_path LIKE '%/CC-MAIN-2014-%';
