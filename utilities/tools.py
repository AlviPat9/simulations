"""

tools
=============
This file defines useful functions thar are available to other projects and that can be used in other calculations.

Author: Alvaro Marcos Canedo
"""

from numpy import cos, sin, sqrt, arctan2


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """

    Function to calculate the distance between two points in the space based on latitude and longitude.

    @param lat1: Latitude of point 1 in the space.
    @type lat1: float
    @param lon1: Longitude of point 1 in the space.
    @type lon1: float
    @param lat2: Latitude of point 2 in the space.
    @type lat2: float
    @param lon2: Longitude of point 2 in the space.
    @type lon2: float

    @return: Actual distance between two points in space
    @rtype: float
    """
    R = 6378137.0
    a = sin((lat1 - lat2) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon1 - lon2) / 2) ** 2
    c = 2 * arctan2(sqrt(a), sqrt(1. - a))

    return R * c


__all__ = ["haversine"]
