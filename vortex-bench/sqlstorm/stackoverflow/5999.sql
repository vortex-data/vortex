WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        u.DisplayName AS AuthorName,
        p.ViewCount,
        p.Score,
        ROW_NUMBER() OVER (PARTITION BY pt.Name ORDER BY p.ViewCount DESC) AS RankByViews,
        ROW_NUMBER() OVER (PARTITION BY pt.Name ORDER BY p.Score DESC) AS RankByScore
    FROM 
        Posts p
    JOIN 
        PostTypes pt ON p.PostTypeId = pt.Id
    JOIN 
        Users u ON p.OwnerUserId = u.Id
    WHERE 
        p.CreationDate > (cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '30 days')
    AND 
        p.ViewCount > 100
)

SELECT 
    rp.PostId,
    rp.Title,
    rp.CreationDate,
    rp.AuthorName,
    rp.ViewCount,
    rp.Score,
    CASE 
        WHEN rp.RankByViews <= 10 THEN 'Top 10 Viewed'
        ELSE 'Other'
    END AS ViewRankCategory,
    CASE 
        WHEN rp.RankByScore <= 10 THEN 'Top 10 Scored'
        ELSE 'Other'
    END AS ScoreRankCategory
FROM 
    RankedPosts rp
WHERE 
    rp.RankByViews <= 10 OR rp.RankByScore <= 10
ORDER BY 
    rp.ViewCount DESC, rp.Score DESC;