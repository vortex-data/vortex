WITH RankedMovies AS (
    SELECT 
        mt.title,
        mt.production_year,
        COUNT(DISTINCT cc.subject_id) AS total_cast,
        ROW_NUMBER() OVER (PARTITION BY mt.production_year ORDER BY COUNT(DISTINCT cc.subject_id) DESC) AS rank
    FROM 
        title mt
    LEFT JOIN 
        complete_cast cc ON mt.id = cc.movie_id
    GROUP BY 
        mt.title, mt.production_year
),
MovieKeywords AS (
    SELECT 
        mt.title,
        mk.keyword,
        ROW_NUMBER() OVER (PARTITION BY mt.id ORDER BY mk.id) AS keyword_rank
    FROM 
        title mt
    JOIN 
        movie_keyword mvk ON mt.id = mvk.movie_id
    JOIN 
        keyword mk ON mvk.keyword_id = mk.id
)
SELECT 
    rm.title,
    rm.production_year,
    rm.total_cast,
    STRING_AGG(mk.keyword, ', ') AS keywords
FROM 
    RankedMovies rm
LEFT JOIN 
    MovieKeywords mk ON mk.title = rm.title
WHERE 
    rm.rank <= 5
GROUP BY 
    rm.title, rm.production_year, rm.total_cast
ORDER BY 
    rm.production_year DESC, rm.total_cast DESC;
