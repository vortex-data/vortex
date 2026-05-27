
WITH RECURSIVE RecursivePostHierarchy AS (
    SELECT 
        Id AS PostId, 
        Title, 
        ParentId,
        CreationDate,
        0 AS Level
    FROM 
        Posts
    WHERE 
        ParentId IS NULL

    UNION ALL

    SELECT 
        p.Id AS PostId,
        p.Title,
        p.ParentId,
        p.CreationDate,
        r.Level + 1
    FROM 
        Posts p
    INNER JOIN 
        RecursivePostHierarchy r ON p.ParentId = r.PostId
),
PostStats AS (
    SELECT 
        p.Id AS PostId,
        COUNT(c.Id) AS CommentCount,
        COUNT(DISTINCT v.UserId) FILTER (WHERE v.VoteTypeId = 2) AS UpvoteCount,
        COUNT(DISTINCT v.UserId) FILTER (WHERE v.VoteTypeId = 3) AS DownvoteCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 WHEN v.VoteTypeId = 3 THEN -1 ELSE 0 END) AS NetVotes,
        MAX(ph.CreationDate) AS LastHistoryUpdate
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    LEFT JOIN 
        PostHistory ph ON p.Id = ph.PostId
    GROUP BY 
        p.Id
),
UserBadges AS (
    SELECT 
        u.Id AS UserId, 
        COUNT(b.Id) FILTER (WHERE b.Class = 1) AS GoldBadges,
        COUNT(b.Id) FILTER (WHERE b.Class = 2) AS SilverBadges,
        COUNT(b.Id) FILTER (WHERE b.Class = 3) AS BronzeBadges,
        SUM(CASE WHEN b.TagBased THEN 1 ELSE 0 END) AS TagBasedBadges
    FROM 
        Users u
    LEFT JOIN 
        Badges b ON u.Id = b.UserId
    GROUP BY 
        u.Id
)
SELECT 
    p.Title AS PostTitle,
    p.CreationDate AS PostCreationDate,
    ps.CommentCount,
    ps.UpvoteCount,
    ps.DownvoteCount,
    ps.NetVotes,
    COALESCE(u.DisplayName, 'Unknown User') AS OwnerDisplayName,
    ub.GoldBadges,
    ub.SilverBadges,
    ub.BronzeBadges,
    ph.Level AS PostLevel,
    ph.ParentId AS ParentPostId,
    CASE 
        WHEN ps.LastHistoryUpdate IS NOT NULL AND ps.LastHistoryUpdate < TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '1 year' 
        THEN 'Stale Post' 
        ELSE 'Active Post' 
    END AS PostStatus
FROM 
    Posts p
LEFT JOIN 
    PostStats ps ON p.Id = ps.PostId
LEFT JOIN 
    Users u ON p.OwnerUserId = u.Id
LEFT JOIN 
    UserBadges ub ON u.Id = ub.UserId
LEFT JOIN 
    RecursivePostHierarchy ph ON p.Id = ph.PostId
WHERE 
    p.CreationDate > TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '1 year'
ORDER BY 
    ps.NetVotes DESC, 
    ps.CommentCount DESC
LIMIT 50;
