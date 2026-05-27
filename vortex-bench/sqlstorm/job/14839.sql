
SELECT 
    t.title AS movie_title,
    a.name AS actor_name,
    ci.nr_order AS actor_order,
    ct.kind AS company_type,
    COUNT(mk.keyword_id) AS keyword_count
FROM 
    title t
JOIN 
    complete_cast cc ON t.id = cc.movie_id
JOIN 
    cast_info ci ON ci.id = cc.subject_id
JOIN 
    aka_name a ON a.person_id = ci.person_id
JOIN 
    movie_companies mc ON mc.movie_id = t.id
JOIN 
    company_type ct ON ct.id = mc.company_type_id
JOIN 
    movie_keyword mk ON mk.movie_id = t.id
GROUP BY 
    t.title, a.name, ci.nr_order, ct.kind
ORDER BY 
    t.title, actor_order;
