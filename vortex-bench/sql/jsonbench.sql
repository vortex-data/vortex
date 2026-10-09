-- JSONBench queries over Bluesky events, numbered from Q0 in file order. The harness splits the
-- file on semicolons, so a comment must never contain one.
--
-- `{str:a.b}` and `{i64:a.b}` read JSON path `a.b` of the `data` column as a string or a 64-bit
-- integer. The harness expands them into each engine's idiom for each storage format.

-- Q0. Top event types.
SELECT {str:commit.collection} AS event, count(*) AS cnt
FROM bluesky
GROUP BY event
ORDER BY cnt DESC;

-- Q1. Top event types together with unique users per event type.
SELECT {str:commit.collection} AS event, count(*) AS cnt, count(DISTINCT {str:did}) AS users
FROM bluesky
WHERE {str:kind} = 'commit' AND {str:commit.operation} = 'create'
GROUP BY event
ORDER BY cnt DESC;

-- Q2. When do people use Bluesky.
SELECT {str:commit.collection} AS event,
       date_part('hour', to_timestamp({i64:time_us} / 1000000)) AS hour_of_day,
       count(*) AS cnt
FROM bluesky
WHERE {str:kind} = 'commit'
  AND {str:commit.operation} = 'create'
  AND {str:commit.collection} IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event;

-- Q3. The three users who posted first.
SELECT {str:did} AS user_id, min({i64:time_us}) AS first_post_us
FROM bluesky
WHERE {str:kind} = 'commit'
  AND {str:commit.operation} = 'create'
  AND {str:commit.collection} = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_us ASC, user_id
LIMIT 3;

-- Q4. The three users with the longest posting activity span.
SELECT {str:did} AS user_id, (max({i64:time_us}) - min({i64:time_us})) / 1000 AS activity_span_ms
FROM bluesky
WHERE {str:kind} = 'commit'
  AND {str:commit.operation} = 'create'
  AND {str:commit.collection} = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span_ms DESC, user_id
LIMIT 3;
