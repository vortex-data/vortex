WITH RankedMovies AS (
    SELECT 
        t.title,
        t.production_year,
        COUNT(DISTINCT ci.person_id) AS total_cast,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY COUNT(DISTINCT ci.person_id) DESC) AS rank
    FROM 
        aka_title t
    LEFT JOIN 
        cast_info ci ON t.id = ci.movie_id
    GROUP BY 
        t.id, t.title, t.production_year
),
TopMovies AS (
    SELECT 
        rm.title, 
        rm.production_year,
        rm.total_cast
    FROM 
        RankedMovies rm
    WHERE 
        rm.rank <= 5
),
FilteredKeywords AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(k.keyword, ', ') AS keyword_list
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
)
SELECT 
    tm.title,
    tm.production_year,
    tm.total_cast,
    COALESCE(fk.keyword_list, 'No Keywords') AS keywords,
    COALESCE(cn.name, 'Undisclosed Company') AS production_company
FROM 
    TopMovies tm
LEFT JOIN 
    movie_companies mc ON tm.production_year = mc.movie_id
LEFT JOIN 
    company_name cn ON mc.company_id = cn.id
LEFT JOIN 
    FilteredKeywords fk ON tm.production_year = fk.movie_id
WHERE 
    tm.total_cast > 0
ORDER BY 
    tm.production_year DESC, 
    tm.total_cast DESC;
