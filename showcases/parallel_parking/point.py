import math


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)

    @staticmethod
    def rotate(point, theta):
        # counter-clockwise; assuming subtracted reference point already
        x = math.cos(theta) * point.x - math.sin(theta) * point.y
        y = math.sin(theta) * point.x + math.cos(theta) * point.y

        return Point2D(x, y)

    def move(self, orientation, distance, forward: bool):
        # move on a straight line with orientation
        if forward:
            self.x -= distance * math.sin(orientation)
            self.y += distance * math.cos(orientation)
        else:
            self.x += distance * math.sin(orientation)
            self.y -= distance * math.cos(orientation)
