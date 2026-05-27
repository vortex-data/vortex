WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS title_rank,
        COUNT(*) OVER (PARTITION BY t.production_year) AS production_count
    FROM 
        aka_title t
    WHERE 
        t.production_year IS NOT NULL
),
ActorRoles AS (
    SELECT 
        c.movie_id,
        COUNT(DISTINCT c.person_id) AS actor_count,
        SUM(CASE WHEN c.note IS NOT NULL THEN 1 ELSE 0 END) AS has_note_count
    FROM 
        cast_info c
    GROUP BY 
        c.movie_id
),
MovieKeywords AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
),
MoviesWithDetails AS (
    SELECT 
        rm.movie_id,
        rm.title,
        rm.production_year,
        COALESCE(ar.actor_count, 0) AS total_actors,
        COALESCE(ar.has_note_count, 0) AS notes_on_actors,
        COALESCE(mk.keywords, 'No Keywords') AS keywords,
        rm.production_count,
        CASE 
            WHEN rm.production_count > 5 THEN 'Popular Year' 
            ELSE 'Less Popular Year' 
        END AS popularity_category
    FROM 
        RankedMovies rm
    LEFT JOIN 
        ActorRoles ar ON rm.movie_id = ar.movie_id
    LEFT JOIN 
        MovieKeywords mk ON rm.movie_id = mk.movie_id
),
FilteredMovies AS (
    SELECT 
        mwd.title,
        mwd.production_year,
        mwd.total_actors,
        mwd.notes_on_actors,
        mwd.keywords,
        mwd.popularity_category
    FROM 
        MoviesWithDetails mwd
    WHERE 
        mwd.total_actors > 3 
        AND mwd.production_year BETWEEN 2000 AND 2023
        AND NOT (mwd.keywords IS NULL OR mwd.keywords = 'No Keywords')
)
SELECT 
    title,
    production_year,
    total_actors,
    notes_on_actors,
    keywords,
    popularity_category
FROM 
    FilteredMovies
ORDER BY 
    production_year DESC, 
    total_actors DESC
LIMIT 10;