WITH RankedMovies AS (
    SELECT 
        mt.id AS movie_id,
        mt.title,
        mt.production_year,
        st.kind AS movie_kind,
        STRING_AGG(DISTINCT cn.name, ', ') AS production_companies,
        COUNT(DISTINCT c.person_id) AS cast_count
    FROM 
        aka_title mt
    JOIN 
        movie_info mi ON mt.id = mi.movie_id
    JOIN 
        movie_companies mc ON mt.id = mc.movie_id
    JOIN 
        company_name cn ON mc.company_id = cn.id
    JOIN 
        cast_info c ON mt.id = c.movie_id
    JOIN 
        kind_type st ON mt.kind_id = st.id
    WHERE 
        mi.info_type_id = (SELECT id FROM info_type WHERE info = 'summary')
    GROUP BY 
        mt.id, mt.title, mt.production_year, st.kind
),
TopRatedMovies AS (
    SELECT 
        movie_id, 
        title, 
        production_year, 
        movie_kind, 
        production_companies, 
        cast_count,
        RANK() OVER (ORDER BY cast_count DESC) AS rank
    FROM 
        RankedMovies
)
SELECT 
    tr.movie_id, 
    tr.title, 
    tr.production_year, 
    tr.movie_kind, 
    tr.production_companies, 
    tr.cast_count
FROM 
    TopRatedMovies tr
WHERE 
    tr.rank <= 10
ORDER BY 
    tr.cast_count DESC;