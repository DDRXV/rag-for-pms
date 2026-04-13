"""
SkillAgents AI mock org chart as a simple adjacency dictionary.

The key is a person. The value is the list of direct reports. This is the
smallest possible graph representation. Real production systems would use
Neo4j or Neptune, but the shape of the routing problem is identical.
"""

ORG_CHART = {
    "Priya Shah (CEO)": [
        "Dan Alvarez (VP Engineering)",
        "Mei Wong (VP Product)",
        "Rahul Iyer (VP Growth)",
        "Sara Kim (VP Operations)",
    ],
    "Dan Alvarez (VP Engineering)": [
        "Leo Park (Director, Platform)",
        "Nisha Rao (Director, ML)",
        "Tomas Weber (Director, Frontend)",
    ],
    "Mei Wong (VP Product)": [
        "Jordan Blake (Director, Learner Experience)",
        "Aisha Okafor (Director, Enterprise Product)",
    ],
    "Rahul Iyer (VP Growth)": [
        "Chloe Martin (Head of Content)",
        "Ethan Brooks (Head of Partnerships)",
    ],
    "Sara Kim (VP Operations)": [
        "Marcus Lee (Head of Support)",
        "Hana Yoshida (Head of Finance)",
    ],
    "Leo Park (Director, Platform)": [
        "Ana Costa (Staff Engineer, Infra)",
        "Vik Patel (Senior Engineer, Billing)",
    ],
    "Nisha Rao (Director, ML)": [
        "Diego Ramos (ML Engineer)",
        "Yuki Tanaka (ML Engineer)",
    ],
}


def direct_reports(person: str) -> list:
    """Return the list of direct reports for a person, or an empty list."""
    return ORG_CHART.get(person, [])


def find_person(name_fragment: str) -> str | None:
    """
    Find a person in the org chart by a case-insensitive substring match on
    their full label. Returns the first match or None. Useful when the router
    hands over a loose reference like "VP Engineering" and you need the exact
    key to look up direct reports.
    """
    needle = name_fragment.lower()
    for person in ORG_CHART.keys():
        if needle in person.lower():
            return person
    # Also scan values so "Leo Park" finds its full label.
    for reports in ORG_CHART.values():
        for r in reports:
            if needle in r.lower():
                return r
    return None
