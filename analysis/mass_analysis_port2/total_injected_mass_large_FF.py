"""
Hardcoded injection profile for the large fluid flower. 
"""


def total_mass(t):
    """
    Returns total injected mass (grams) as a function of time (minutes).

    NOTE: Many numbers can be added together, but was implemented like
    this for readibility of the developer.
    """
    if t < 135:
        return 0
    elif t < 135 + 4.5:
        return 1.82 * 10 / 2 * (t-135) / 1000
    elif t < 300:
        return (
            1.82 * 10 / 2 * 4.5 / 1000
            + 1.82 * 10 * (t - 135 - 4.5) / 1000
        )
    elif t < 300 + 4.5:
        return (
            1.82 * 10 / 2 * 4.5 / 1000
            + 1.82 * 10 * (300 - 135 - 4.5) / 1000
            + 1.82 * 10 / 2 * (t - 300) / 1000
        )
    else:
        return (
            1.82 * 10 * 4.5 / 1000
            + 1.82 * 10 * (300 - 135 - 4.5) / 1000
        )
