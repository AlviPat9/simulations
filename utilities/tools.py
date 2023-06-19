"""

tools
=============
This file defines useful functions thar are available to other projects and that can be used in other calculations.

Author: Alvaro Marcos Canedo
"""

from numpy import cos, sin, sqrt, arctan2, ndarray, array, concatenate


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


def keep_angle_range(angle: float, min_angle: float, max_angle: float) -> ndarray:
    """

    Function to keep angle in a specified range.

    @param angle: Angle to keep in specified range
    @type angle: float
    @param min_angle: Minimum value the angle can have.
    @type min_angle: float
    @param max_angle: Maximum value the angle can have.
    @type max_angle: float

    @return: Angle in specified range.
    @rtype: float
    """

    range = max_angle - min_angle

    normalize_angle = (angle - min_angle) % range

    return normalize_angle + min_angle


def convert_dict_to_ndarray(data: dict) -> ndarray:
    """

    Function to convert a dictionary to a numpy array format.

    @param data: Data included in dictionary to convert to a numpy array.
    @type data: dict

    @return: Dictionary converted to a numpy array.
    @rtype: ndarray
    """

    # Extract values from the dictionary.
    dict_values = list(data.values())

    # Convert numbers to numpy arrays.
    dict_values = [array(val) if not isinstance(val, ndarray) else val for val in dict_values]

    # Concatenate into a new numpy array and return the value
    return concatenate(dict_values)


__all__ = ["haversine", "keep_angle_range", "convert_dict_to_ndarray"]
