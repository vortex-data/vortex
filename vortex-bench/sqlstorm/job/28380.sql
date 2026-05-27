
WITH ActorMovies AS (
    SELECT
        a.id AS actor_id,
        a.name AS actor_name,
        m.title AS movie_title,
        m.production_year,
        r.role AS actor_role
    FROM
        aka_name a
    JOIN cast_info ci ON a.person_id = ci.person_id
    JOIN title m ON ci.movie_id = m.id
    JOIN role_type r ON ci.role_id = r.id
),
MovieKeywords AS (
    SELECT
        m.id AS movie_id,
        k.keyword AS movie_keyword
    FROM
        title m
    JOIN movie_keyword mk ON m.id = mk.movie_id
    JOIN keyword k ON mk.keyword_id = k.id
),
ActorKeywords AS (
    SELECT
        am.actor_id,
        am.actor_name,
        am.movie_title,
        am.production_year,
        STRING_AGG(mk.movie_keyword, ', ') AS keywords
    FROM
        ActorMovies am
    LEFT JOIN MovieKeywords mk ON am.movie_title = mk.movie_keyword
    GROUP BY
        am.actor_id, am.actor_name, am.movie_title, am.production_year
)
SELECT
    ak.actor_id,
    ak.actor_name,
    ak.movie_title,
    ak.production_year,
    ak.keywords
FROM
    ActorKeywords ak
WHERE
    ak.keywords IS NOT NULL
ORDER BY
    ak.production_year DESC, ak.actor_name;
