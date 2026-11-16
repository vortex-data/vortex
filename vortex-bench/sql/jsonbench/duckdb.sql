-- Q0: events per collection.
SELECT data.commit.collection::VARCHAR AS event,
       COUNT(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;

-- Q1: events and distinct users per collection, for created commits.
SELECT data.commit.collection::VARCHAR AS event,
       COUNT(*) AS count,
       COUNT(DISTINCT data.did::VARCHAR) AS users
FROM bluesky
WHERE data.kind::VARCHAR = 'commit'
  AND data.commit.operation::VARCHAR = 'create'
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;

-- Q2: posts, reposts and likes per hour of the day.
SELECT data.commit.collection::VARCHAR AS event,
       EXTRACT(HOUR FROM make_timestamp(data.time_us::BIGINT)) AS hour_of_day,
       COUNT(*) AS count
FROM bluesky
WHERE data.kind::VARCHAR = 'commit'
  AND data.commit.operation::VARCHAR = 'create'
  AND data.commit.collection::VARCHAR IN (
      'app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like'
  )
GROUP BY event, hour_of_day
ORDER BY hour_of_day ASC NULLS FIRST, event;

-- Q3: the three users who posted first.
SELECT data.did::VARCHAR AS user_id,
       MIN(make_timestamp(data.time_us::BIGINT)) AS first_post_ts
FROM bluesky
WHERE data.kind::VARCHAR = 'commit'
  AND data.commit.operation::VARCHAR = 'create'
  AND data.commit.collection::VARCHAR = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;

-- Q4: the three users with the longest span between their first and last post, in milliseconds.
-- Each endpoint is divided before subtracting, to count millisecond boundaries as ClickHouse's
-- dateDiff does.
SELECT data.did::VARCHAR AS user_id,
       MAX(data.time_us::BIGINT) // 1000 - MIN(data.time_us::BIGINT) // 1000 AS activity_span
FROM bluesky
WHERE data.kind::VARCHAR = 'commit'
  AND data.commit.operation::VARCHAR = 'create'
  AND data.commit.collection::VARCHAR = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
