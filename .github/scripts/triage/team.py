"""Who counts as the Vortex team when classifying GitHub activity."""

# GitHub logins, in lowercase, seeded from the vortex-data/committers and
# vortex-data/contributors teams. Everyone else is treated as an external contributor.
TEAM = frozenset(
    {
        "0ax1",
        "a10y",
        "adamgs",
        "adriangb",
        "asubiotto",
        "blaginin",
        "brancz",
        "connortsui20",
        "danking",
        "delta003",
        "gargiulofrancesco",
        "gatesn",
        "harukimoriarty",
        "joseph-isaacs",
        "lorenzhs",
        "lwwmanning",
        "mhk197",
        "moshap-firebolt",
        "mprammer",
        "myrrc",
        "onursatici",
        "palaska",
        "robert3005",
        "thorfour",
        "wesm",
        "whelanboyd",
        "xiangpenghao",
        "xinyuzeng",
    }
)
