WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        COUNT(DISTINCT c.person_id) AS num_actors,
        STRING_AGG(DISTINCT a.name, ', ') AS actor_names
    FROM 
        aka_title t
    JOIN 
        complete_cast cc ON t.id = cc.movie_id
    JOIN 
        cast_info c ON cc.subject_id = c.id
    JOIN 
        aka_name a ON c.person_id = a.person_id
    WHERE 
        t.production_year >= 2000
    GROUP BY 
        t.id, t.title, t.production_year
),
TopMovies AS (
    SELECT 
        rm.movie_id,
        rm.title,
        rm.production_year,
        rm.num_actors,
        rm.actor_names,
        RANK() OVER (ORDER BY rm.num_actors DESC) AS actor_rank
    FROM 
        RankedMovies rm
)
SELECT 
    tm.movie_id,
    tm.title,
    tm.production_year,
    tm.num_actors,
    tm.actor_names
FROM 
    TopMovies tm
WHERE 
    tm.actor_rank <= 10
ORDER BY 
    tm.num_actors DESC;
