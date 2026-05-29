WITH RecentPosts AS (
    SELECT 
        p.Id,
        p.Title,
        p.CreationDate,
        p.ViewCount,
        p.Score,
        p.AnswerCount,
        p.CommentCount,
        u.DisplayName AS OwnerName,
        CASE 
            WHEN p.ClosedDate IS NOT NULL THEN 'Closed'
            ELSE 'Open'
        END AS PostStatus
    FROM 
        Posts p
    JOIN 
        Users u ON p.OwnerUserId = u.Id
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '30 days'
),
PostStatistics AS (
    SELECT 
        rp.Id,
        rp.Title,
        rp.CreationDate,
        rp.ViewCount,
        rp.Score,
        rp.AnswerCount,
        rp.CommentCount,
        rp.OwnerName,
        rp.PostStatus,
        ROW_NUMBER() OVER (PARTITION BY rp.PostStatus ORDER BY rp.Score DESC) AS Rank
    FROM 
        RecentPosts rp
),
TopPosts AS (
    SELECT * 
    FROM PostStatistics
    WHERE Rank <= 5
),
PostVoteCounts AS (
    SELECT 
        p.Id AS PostId,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes,
        SUM(CASE WHEN v.VoteTypeId = 5 THEN 1 ELSE 0 END) AS Favorites
    FROM 
        Posts p
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    GROUP BY 
        p.Id
)
SELECT 
    tp.Title,
    tp.CreationDate,
    tp.ViewCount,
    tp.Score,
    tp.AnswerCount,
    tp.CommentCount,
    tp.OwnerName,
    tp.PostStatus,
    COALESCE(v.UpVotes, 0) AS TotalUpVotes,
    COALESCE(v.DownVotes, 0) AS TotalDownVotes,
    COALESCE(v.Favorites, 0) AS TotalFavorites,
    CASE 
        WHEN v.UpVotes IS NOT NULL AND v.DownVotes IS NOT NULL THEN 
            (CAST(v.UpVotes AS decimal) / NULLIF((v.UpVotes + v.DownVotes), 0)) * 100 
        ELSE 0 
    END AS UpVotePercentage
FROM 
    TopPosts tp
LEFT JOIN 
    PostVoteCounts v ON tp.Id = v.PostId
ORDER BY 
    tp.PostStatus, tp.Score DESC;