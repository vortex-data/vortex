WITH RecursivePostStats AS (
    SELECT
        p.Id AS PostId,
        p.Score,
        p.ViewCount,
        p.AnswerCount,
        p.CommentCount,
        p.CreationDate,
        p.OwnerUserId,
        ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.CreationDate DESC) AS rn
    FROM Posts p
    WHERE p.PostTypeId = 1  
),
UserBadges AS (
    SELECT
        b.UserId,
        COUNT(CASE WHEN b.Class = 1 THEN 1 END) AS GoldBadges,
        COUNT(CASE WHEN b.Class = 2 THEN 1 END) AS SilverBadges,
        COUNT(CASE WHEN b.Class = 3 THEN 1 END) AS BronzeBadges
    FROM Badges b
    GROUP BY b.UserId
),
VoteCounts AS (
    SELECT
        v.PostId,
        COUNT(CASE WHEN v.VoteTypeId = 2 THEN 1 END) AS UpVotes,
        COUNT(CASE WHEN v.VoteTypeId = 3 THEN 1 END) AS DownVotes
    FROM Votes v
    GROUP BY v.PostId
),
PostHistoryAnalysis AS (
    SELECT
        ph.PostId,
        MAX(ph.CreationDate) AS LastActivityDate,
        COUNT(DISTINCT CASE WHEN ph.PostHistoryTypeId = 10 THEN ph.UserId END) AS CloseVotes,
        COUNT(DISTINCT CASE WHEN ph.PostHistoryTypeId = 11 THEN ph.UserId END) AS ReopenVotes
    FROM PostHistory ph
    GROUP BY ph.PostId
)
SELECT
    p.Id AS PostId,
    p.Title,
    p.CreationDate,
    p.Score,
    ps.ViewCount,
    ps.AnswerCount,
    ps.CommentCount,
    ps.OwnerUserId,
    u.DisplayName AS OwnerDisplayName,
    COALESCE(ub.GoldBadges, 0) AS GoldBadges,
    COALESCE(ub.SilverBadges, 0) AS SilverBadges,
    COALESCE(ub.BronzeBadges, 0) AS BronzeBadges,
    COALESCE(vc.UpVotes, 0) - COALESCE(vc.DownVotes, 0) AS NetVotes,
    pha.LastActivityDate,
    pha.CloseVotes,
    pha.ReopenVotes
FROM Posts p
JOIN RecursivePostStats ps ON p.Id = ps.PostId
JOIN Users u ON p.OwnerUserId = u.Id
LEFT JOIN UserBadges ub ON u.Id = ub.UserId
LEFT JOIN VoteCounts vc ON p.Id = vc.PostId
LEFT JOIN PostHistoryAnalysis pha ON p.Id = pha.PostId
WHERE ps.rn = 1
  AND p.CreationDate >= (cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year')
ORDER BY NetVotes DESC, p.CreationDate DESC
LIMIT 100;