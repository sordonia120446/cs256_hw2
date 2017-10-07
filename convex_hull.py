"""
Convex hull calculation.

:author Sam O <samuel.ordonia@gmail.com>
"""

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '(x, y) = ({x}, {y})'.format(
            x=self.x,
            y=self.y
        )


def ccw(p1, p2, p3):
    """
    Determine if three points are in a counter-clockwise turn or not.
    Counter-clockwise if ccw > 0
    Clockwise if ccw < 0
    Co-linear if ccw = 0

    Coincidentally, this is the cross-product operation.

    :returns ccw
    """
    return (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x)


def calc_lowest_point(points):
    p0 = points[0]
    for p in points:
        if p.y < p0.y:
            p0 = p

    return Point(p0.x, p0.y)


def calc_leftmost_point(points):
    p0 = points[0]
    for p in points:
        if p.x < p0.x:
            p0 = p

    return Point(p0.x, p0.y)


def convex_hull(points):

    # Triangles/Lines/Points are convex hulls by default
    if len(points) < 4:
        return points

    # Find leftmost point (lowest x)
    p0 = calc_leftmost_point(points)

    def slope(p):
        """
        Used to sort the points relative to a reference.
        Reference can be the leftmost (lowest x) or lowest (lowest y),
        depending on the algorithm approach.

        :returns type float: slope, or 0 if slope is vertical (Inf)
        """
        try:
            return float(p.y - p0.y)/(p.x - p0.x)
        except ZeroDivisionError:
            return 0
    
    # Sort based on the slope relative to the reference point p0
    ps_sorted = sorted(points, key=slope)

    # Build lower hull
    lower = []
    for p in ps_sorted:
        while len(lower) > 1 and ccw(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(ps_sorted):
        while len(upper) > 1 and ccw(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Combine the lower and upper hulls.
    # Omit the last point in each list (repeats)
    convex_hull = lower[:-1] + upper[:-1]

    return convex_hull


if __name__ == '__main__':
    points = []
    points.append(Point(0, 0))
    points.append(Point(1, 0.5))
    points.append(Point(1, -0.5))
    points.append(Point(2, 5))
    points.append(Point(2, -5))
    points.append(Point(3, 0))

    print(f'These are the exercise points:  {points}')

    ch = convex_hull(points)
    print(f'The ch is {ch}')

