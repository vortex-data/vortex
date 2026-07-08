-- GitHub Archive queries over nested event data, numbered from Q0 in file order. The
-- harness splits the file on semicolons, so a comment must never contain one.

-- Q0.
select count(*) from events where payload.ref = 'refs/heads/main';

-- Q1.
select distinct repo.name from events where repo.name like 'spiraldb/%';

-- Q2.
select distinct org.id as org_id from events order by org_id limit 100;

-- Q3.
select actor.login, count() as freq from events group by actor.login order by freq desc limit 10;

-- Q4.
select actor.avatar_url from events where actor.login = 'renovate[bot]';
