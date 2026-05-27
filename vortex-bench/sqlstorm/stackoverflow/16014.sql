SELECT u.DisplayName, COUNT(p.Id) AS PostCount, SUM(v.BountyAmount) AS TotalBounty
FROM Users u
JOIN Posts p ON u.Id = p.OwnerUserId
LEFT JOIN Votes v ON p.Id = v.PostId AND v.VoteTypeId = 8
WHERE u.Reputation > 1000
GROUP BY u.DisplayName
ORDER BY PostCount DESC
LIMIT 10;
