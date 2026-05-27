WITH RankedMovies AS (
    SELECT 
        t.title,
        t.production_year,
        COUNT(ci.person_id) AS total_cast,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY COUNT(ci.person_id) DESC) AS rank
    FROM 
        title t
    LEFT JOIN 
        complete_cast cc ON t.id = cc.movie_id
    LEFT JOIN 
        cast_info ci ON cc.subject_id = ci.movie_id
    GROUP BY 
        t.title, t.production_year
),
FilteredMovies AS (
    SELECT 
        title,
        production_year,
        total_cast
    FROM 
        RankedMovies
    WHERE 
        rank <= 10
),
MovieKeywords AS (
    SELECT 
        t.title,
        k.keyword
    FROM 
        title t
    JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
),
MoviesWithKeywords AS (
    SELECT 
        fm.title,
        fm.production_year,
        fm.total_cast,
        STRING_AGG(mk.keyword, ', ') AS keywords
    FROM 
        FilteredMovies fm
    LEFT JOIN 
        MovieKeywords mk ON fm.title = mk.title
    GROUP BY 
        fm.title, fm.production_year, fm.total_cast
)
SELECT 
    mwk.title,
    mwk.production_year,
    mwk.total_cast,
    COALESCE(mwk.keywords, 'No Keywords') AS keywords,
    CASE 
        WHEN mwk.total_cast > 100 THEN 'Blockbuster'
        WHEN mwk.total_cast BETWEEN 50 AND 100 THEN 'Moderate Hit'
        WHEN mwk.total_cast < 50 THEN 'Flop'
        ELSE 'Unknown' 
    END AS movie_performance
FROM 
    MoviesWithKeywords mwk
WHERE 
    mwk.production_year >= 2000
ORDER BY 
    mwk.total_cast DESC
LIMIT 20;
