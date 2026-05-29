
WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        u.DisplayName AS OwnerName,
        p.CreationDate,
        p.Score,
        COUNT(DISTINCT c.Id) AS CommentCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVoteCount,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVoteCount,
        ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC) AS Rank
    FROM 
        Posts p
    JOIN 
        Users u ON p.OwnerUserId = u.Id
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    WHERE 
        p.CreationDate >= TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '1 year'
    GROUP BY 
        p.Id, p.Title, u.DisplayName, p.CreationDate, p.Score, p.PostTypeId
),
TopPosts AS (
    SELECT 
        rp.PostId,
        rp.Title,
        rp.OwnerName,
        rp.CreationDate,
        rp.Score,
        rp.CommentCount,
        rp.UpVoteCount,
        rp.DownVoteCount
    FROM 
        RankedPosts rp
    WHERE 
        rp.Rank <= 10
)
SELECT 
    tp.Title,
    tp.OwnerName,
    tp.CreationDate,
    tp.Score,
    tp.CommentCount,
    tp.UpVoteCount,
    tp.DownVoteCount,
    COALESCE(b.Name, 'No Badge') AS BadgeName
FROM 
    TopPosts tp
LEFT JOIN 
    Badges b ON tp.PostId = b.UserId
ORDER BY 
    tp.Score DESC, tp.CommentCount DESC;
