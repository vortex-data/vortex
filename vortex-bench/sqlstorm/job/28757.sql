
WITH MovieStats AS (
    SELECT 
        a.title AS MovieTitle,
        a.production_year AS ProductionYear,
        COUNT(DISTINCT c.person_id) AS CastCount,
        STRING_AGG(DISTINCT ak.name, ', ') AS Actors,
        STRING_AGG(DISTINCT kw.keyword, ', ') AS Keywords
    FROM 
        aka_title a
    JOIN 
        complete_cast cc ON a.id = cc.movie_id
    JOIN 
        cast_info c ON cc.subject_id = c.id
    JOIN 
        aka_name ak ON c.person_id = ak.person_id
    LEFT JOIN 
        movie_keyword mw ON a.id = mw.movie_id
    LEFT JOIN 
        keyword kw ON mw.keyword_id = kw.id
    WHERE 
        a.production_year >= 2000
    GROUP BY 
        a.title, a.production_year
), TopMovies AS (
    SELECT 
        MovieTitle,
        ProductionYear,
        CastCount,
        Actors,
        Keywords,
        RANK() OVER (ORDER BY CastCount DESC) AS Rank
    FROM 
        MovieStats
)
SELECT 
    MovieTitle,
    ProductionYear,
    CastCount,
    Actors,
    Keywords
FROM 
    TopMovies
WHERE 
    Rank <= 10
ORDER BY 
    ProductionYear DESC, CastCount DESC;
