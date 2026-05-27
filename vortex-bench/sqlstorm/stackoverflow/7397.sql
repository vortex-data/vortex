WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.ViewCount,
        p.Score,
        COUNT(c.Id) AS CommentCount,
        ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC, p.ViewCount DESC) AS Rank
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year'
    GROUP BY 
        p.Id, p.Title, p.ViewCount, p.Score, p.PostTypeId
),
TopPosts AS (
    SELECT 
        rp.PostId,
        rp.Title,
        rp.ViewCount,
        rp.Score,
        rp.CommentCount,
        CASE 
            WHEN rp.Rank <= 10 THEN 'Top 10'
            ELSE 'Other'
        END AS RankingCategory
    FROM 
        RankedPosts rp
)
SELECT 
    tp.RankingCategory,
    AVG(tp.ViewCount) AS AvgViewCount,
    SUM(tp.CommentCount) AS TotalComments,
    SUM(tp.Score) AS TotalScore,
    COUNT(tp.PostId) AS TotalPosts
FROM 
    TopPosts tp
GROUP BY 
    tp.RankingCategory
ORDER BY 
    tp.RankingCategory DESC;