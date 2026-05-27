WITH RecentPosts AS (
    SELECT 
        p.Id,
        p.Title,
        p.CreationDate,
        p.ViewCount,
        u.DisplayName AS OwnerName,
        COUNT(c.Id) AS CommentCount,
        COUNT(DISTINCT v.UserId) AS VoteCount
    FROM 
        Posts p
        LEFT JOIN Users u ON p.OwnerUserId = u.Id
        LEFT JOIN Comments c ON p.Id = c.PostId
        LEFT JOIN Votes v ON p.Id = v.PostId AND v.VoteTypeId = 2 
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '30 days'
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.ViewCount, u.DisplayName
),
PostHistoryInfo AS (
    SELECT 
        ph.PostId,
        STRING_AGG(DISTINCT pht.Name, ', ') AS EditHistory,
        MAX(ph.CreationDate) AS LastEditDate
    FROM 
        PostHistory ph
        JOIN PostHistoryTypes pht ON ph.PostHistoryTypeId = pht.Id
    GROUP BY 
        ph.PostId
),
TopPosts AS (
    SELECT 
        rp.*, 
        COALESCE(phe.EditHistory, 'No edits') AS EditHistory,
        phe.LastEditDate
    FROM 
        RecentPosts rp
        LEFT JOIN PostHistoryInfo phe ON rp.Id = phe.PostId
    WHERE 
        rp.ViewCount > (SELECT AVG(ViewCount) FROM RecentPosts) 
)
SELECT 
    tp.Id,
    tp.Title,
    tp.CreationDate,
    tp.ViewCount,
    tp.OwnerName,
    tp.CommentCount,
    tp.VoteCount,
    tp.EditHistory,
    tp.LastEditDate,
    CASE 
        WHEN tp.CommentCount > 10 THEN 'High Engagement'
        WHEN tp.CommentCount BETWEEN 5 AND 10 THEN 'Moderate Engagement'
        ELSE 'Low Engagement'
    END AS EngagementLevel
FROM 
    TopPosts tp
ORDER BY 
    tp.ViewCount DESC, 
    tp.CreationDate ASC
LIMIT 100;