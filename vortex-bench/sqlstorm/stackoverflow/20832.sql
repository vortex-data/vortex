WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.ViewCount,
        p.Score,
        p.CreationDate,
        ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.ViewCount DESC, p.Score DESC) AS Rank,
        COALESCE(p.AcceptedAnswerId, -1) AS AnswerStatus
    FROM 
        Posts p
    WHERE 
        p.CreationDate >= cast('2024-10-01' as date) - INTERVAL '1 year'
        AND p.ViewCount IS NOT NULL
),
PostStatistics AS (
    SELECT 
        u.Id AS UserId,
        SUM(CASE WHEN p.AnswerCount > 0 THEN 1 ELSE 0 END) AS TotalQuestionsAnswered,
        COUNT(DISTINCT p.Id) AS TotalPosts,
        AVG(p.Score) AS AverageScore,
        SUM(CASE WHEN b.Class = 1 THEN 1 ELSE 0 END) AS GoldBadges,
        SUM(CASE WHEN b.Class = 2 THEN 1 ELSE 0 END) AS SilverBadges,
        SUM(CASE WHEN b.Class = 3 THEN 1 ELSE 0 END) AS BronzeBadges
    FROM 
        Users u
    LEFT JOIN 
        Posts p ON u.Id = p.OwnerUserId
    LEFT JOIN 
        Badges b ON u.Id = b.UserId
    GROUP BY 
        u.Id
),
ClosedPostReasons AS (
    SELECT 
        ph.PostId,
        COUNT(*) AS CloseVoteCount,
        STRING_AGG(CASE WHEN ph.PostHistoryTypeId = 10 THEN cr.Name END, ', ') AS CloseReasons
    FROM 
        PostHistory ph
    LEFT JOIN 
        CloseReasonTypes cr ON cr.Id::text = ph.Comment
    GROUP BY 
        ph.PostId
),
UserPostLinkages AS (
    SELECT 
        pl.PostId,
        pl.RelatedPostId,
        COUNT(pl.Id) AS LinkCount
    FROM 
        PostLinks pl
    JOIN 
        Posts p ON pl.PostId = p.Id
    WHERE 
        p.CreationDate < cast('2024-10-01' as date) - INTERVAL '6 months'
    GROUP BY 
        pl.PostId, pl.RelatedPostId
),
FinalStats AS (
    SELECT 
        ps.UserId,
        ps.TotalQuestionsAnswered,
        ps.TotalPosts,
        ps.AverageScore,
        ps.GoldBadges,
        ps.SilverBadges,
        ps.BronzeBadges,
        COALESCE(rp.PostId, 0) AS TopPostId,
        COALESCE(rp.Title, 'No Trending Post') AS TopPostTitle,
        COALESCE(rp.ViewCount, 0) AS TopPostViewCount,
        COALESCE(rp.Score, 0) AS TopPostScore,
        COALESCE(cpr.CloseVoteCount, 0) AS TotalCloseVotes,
        COALESCE(cpr.CloseReasons, 'No Close Reasons') AS CloseReasons,
        COALESCE(pl.LinkCount, 0) AS TotalRelatedLinks
    FROM 
        PostStatistics ps
    LEFT JOIN 
        RankedPosts rp ON ps.UserId = rp.PostId
    LEFT JOIN 
        ClosedPostReasons cpr ON rp.PostId = cpr.PostId
    LEFT JOIN 
        UserPostLinkages pl ON ps.UserId = pl.PostId
)
SELECT 
    *,
    CASE 
        WHEN TotalPosts = 0 THEN 'No activity'
        ELSE 'Active User'
    END AS UserActivityStatus
FROM 
    FinalStats
WHERE 
    TotalQuestionsAnswered > 5
    AND GoldBadges > 0
ORDER BY 
    TotalPosts DESC,
    AverageScore DESC;