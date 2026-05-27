
WITH RankedBadges AS (
    SELECT 
        b.UserId,
        b.Name,
        b.Class,
        RANK() OVER (PARTITION BY b.UserId ORDER BY b.Date DESC) AS BadgeRank
    FROM 
        Badges b
),
PostInformation AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.OwnerUserId,
        COUNT(c.Id) AS CommentCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes,
        MAX(bh.CreationDate) AS LastEditDate
    FROM 
        Posts p
        LEFT JOIN Comments c ON p.Id = c.PostId
        LEFT JOIN Votes v ON p.Id = v.PostId
        LEFT JOIN PostHistory bh ON p.Id = bh.PostId AND bh.PostHistoryTypeId IN (4, 5)
    WHERE 
        p.CreationDate > CURRENT_TIMESTAMP - INTERVAL '1 year'
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.OwnerUserId
),
UserStatistics AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        COALESCE(MAX(b.Name), 'No Badges') AS BestBadge,
        COUNT(DISTINCT pi.PostId) AS PostCount,
        SUM(COALESCE(pi.CommentCount, 0)) AS TotalComments,
        SUM(pi.UpVotes) AS TotalUpVotes,
        SUM(pi.DownVotes) AS TotalDownVotes
    FROM 
        Users u
        LEFT JOIN RankedBadges b ON u.Id = b.UserId AND b.BadgeRank = 1
        LEFT JOIN PostInformation pi ON u.Id = pi.OwnerUserId
    GROUP BY 
        u.Id, u.DisplayName
)
SELECT 
    us.UserId,
    us.DisplayName,
    us.BestBadge,
    us.PostCount,
    us.TotalComments,
    us.TotalUpVotes,
    us.TotalDownVotes,
    CASE 
        WHEN us.PostCount > 10 THEN 'Active'
        ELSE 'Less Active'
    END AS ActivityStatus,
    NULLIF(us.TotalUpVotes - us.TotalDownVotes, 0) AS VoteBalance
FROM 
    UserStatistics us
WHERE 
    us.TotalComments > 0 OR us.PostCount > 0
ORDER BY 
    us.TotalUpVotes DESC, us.TotalComments DESC
LIMIT 20;
