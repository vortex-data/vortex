-- JSONBench (https://github.com/ClickHouse/JSONBench) over a Variant column `data`.
-- Q0: Top event types
SELECT variant_get(data, 'commit.collection', 'Utf8View') AS event, count(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC;

-- Q1: Top event types together with unique users per event type
SELECT
    variant_get(data, 'commit.collection', 'Utf8View') AS event,
    count(*) AS count,
    count(DISTINCT variant_get(data, 'did', 'Utf8View')) AS users
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
GROUP BY event
ORDER BY count DESC;

-- Q2: When do people use BlueSky
SELECT
    variant_get(data, 'commit.collection', 'Utf8View') AS event,
    date_part('hour', to_timestamp_micros(variant_get(data, 'time_us', 'Int64'))) AS hour_of_day,
    count(*) AS count
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event;

-- Q3: top 3 post veterans
SELECT
    variant_get(data, 'did', 'Utf8View') AS user_id,
    min(to_timestamp_micros(variant_get(data, 'time_us', 'Int64'))) AS first_post_ts
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC
LIMIT 3;

-- Q4: top 3 users with longest activity, in milliseconds
SELECT
    variant_get(data, 'did', 'Utf8View') AS user_id,
    (max(variant_get(data, 'time_us', 'Int64')) - min(variant_get(data, 'time_us', 'Int64'))) / 1000 AS activity_span
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC
LIMIT 3;
