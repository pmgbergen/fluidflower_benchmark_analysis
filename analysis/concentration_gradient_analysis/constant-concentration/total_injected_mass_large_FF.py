"""
Injection profile for the large fluid flower as recorded by the
mass controller at injection.

The CO2 has been injected through two injection ports with
separate injection protocols.
"""


def total_mass_co2_port1(t: float) -> float:
    """
    Returns total injected CO2 mass (grams) injected through port 1,
    as a function of time (minutes).

    NOTE: Many numbers can be added together, but was implemented like
    this for readibility of the developer.

    Args:
        t (float): time in minutes.

    Returns:
        float: total CO2 mass injected in port 2.
    """
    if t < 4.5:
        return 1.82 * 10 / 2 * t / 1000

    elif t < 300:
        return 1.82 * 10 / 2 * 4.5 / 1000 + 1.82 * 10 * (t - 4.5) / 1000

    elif t < 300 + 4.5:
        return (
            1.82 * 10 / 2 * 4.5 / 1000
            + 1.82 * 10 * (300 - 4.5) / 1000
            + 1.82 * 10 / 2 * (t - 300) / 1000
        )
    else:
        return 1.82 * 10 * 4.5 / 1000 + 1.82 * 10 * (300 - 4.5) / 1000


def total_mass_co2_port2(t: float) -> float:
    """
    Returns total injected CO2 mass (grams) injected through port 2,
    as a function of time (minutes).

    NOTE: Many numbers can be added together, but was implemented like
    this for readibility of the developer.

    Args:
        t (float): time in minutes.

    Returns:
        float: total CO2 mass injected in port 2.
    """
    if t < 135:
        return 0
    elif t < 135 + 4.5:
        return 1.82 * 10 / 2 * (t - 135) / 1000
    elif t < 300:
        return 1.82 * 10 / 2 * 4.5 / 1000 + 1.82 * 10 * (t - 135 - 4.5) / 1000
    elif t < 300 + 4.5:
        return (
            1.82 * 10 / 2 * 4.5 / 1000
            + 1.82 * 10 * (300 - 135 - 4.5) / 1000
            + 1.82 * 10 / 2 * (t - 300) / 1000
        )
    else:
        return 1.82 * 10 * 4.5 / 1000 + 1.82 * 10 * (300 - 135 - 4.5) / 1000
