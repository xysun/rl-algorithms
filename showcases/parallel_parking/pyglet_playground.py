import math

import pyglet

'''
pyglet works with opengl without any other libs

We want to:
- render a window, showing 3 cars (2 boxes, 1 with wheel and controllable by keyboard and external data feed)
- controls: steering (constant angular velocity), driving: True/False, forward: True/False
assume constant velocity for both angular and motion

TODO: 
- [DONE] R should be calculated from top wheel position, instead of top_left/top_right (right now we reduce turning_radius as a crude approx)
- [DONE] update steering continuously while moving (pyglet's key "motion")
- tests & refactoring
- to avoid errors between meter and pixels, make them different types
- collision detection
- parking success detection
- openai gym integration
- [OPTIONAL] draw fancy cars
'''

w = 480
h = 640

# file:///Users/xiayunsun/Downloads/golf-vii-pa-dimensions.pdf
golf_h = 4.3
golf_w = 2
golf_wheelbase = 2.6
golf_turning_radius = 9 / 2  # adjust for things like we do not model the wheels exactly

car_h = 160
pixel_to_meter = car_h / golf_h
car_w = int(golf_w * pixel_to_meter)

offset = int(0.2 * pixel_to_meter)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def rotate(self, theta):
        # counter-clockwise; assuming subtracted reference point already
        x = math.cos(theta) * self.x - math.sin(theta) * self.y
        y = math.sin(theta) * self.x + math.cos(theta) * self.y
        return Point(x, y)


class CarControl:
    def __init__(self):
        self.drive = False
        self.forward = True
        self.psi = 0  # in degrees, wheel pointing left is positive

    def __str__(self):
        return "driving: %s, forward: %s, turning: %s" % (self.drive, self.forward, self.psi)


def drive_straight(point: Point, orientation, distance, forward: bool):
    if forward:
        return Point(
            point.x - distance * math.sin(orientation),
            point.y + distance * math.cos(orientation)
        )
    else:
        return Point(
            point.x + distance * math.sin(orientation),
            point.y - distance * math.cos(orientation)
        )


class CarState:
    # everything in pixel
    def __init__(self):
        self._w = car_w
        self._h = car_h
        self._wheel_base = golf_wheelbase * pixel_to_meter

        self.x = 3 * offset + 1.5 * self._w
        self.y = offset + 3.5 * self._h

        self.center = Point(self.x, self.y)

        self.v_in_meters = 2 * 1600 / 3600.  # 5 miles per hour
        self.v = car_h / (golf_h / self.v_in_meters)
        print("velocity: ", self.v_in_meters, self.v)
        self.orientation = 0  # radius, left is positive

        # anchor positions, offsets are constant
        self.bottom_left_offset = Point(- 0.5 * self._w, - 0.5 * self._h)
        self.top_left_offset = Point(- 0.5 * self._w, + 0.5 * self._h)
        self.top_right_offset = Point(+ 0.5 * self._w, + 0.5 * self._h)
        self.bottom_right_offset = Point(0.5 * self._w, - 0.5 * self._h)

        self.bottom_left = self.center + self.bottom_left_offset
        self.top_left = self.center + self.top_left_offset
        self.top_right = self.center + self.top_right_offset
        self.bottom_right = self.center + self.bottom_right_offset

        self.last_psi = None
        self.turning_center = None

    def from_center(self):
        self.bottom_left = self.bottom_left_offset.rotate(self.orientation) + self.center
        self.top_left = self.top_left_offset.rotate(self.orientation) + self.center
        self.top_right = self.top_right_offset.rotate(self.orientation) + self.center
        self.bottom_right = self.bottom_right_offset.rotate(self.orientation) + self.center

    def update_vertices(self, control: CarControl, dt=None):

        # straight-line driving
        if control.drive:
            if control.psi == 0:
                if self.orientation == 0:
                    if control.forward:
                        y_delta = self.v * dt
                    else:
                        y_delta = -1 * self.v * dt
                    self.center.y += y_delta
                    self.from_center()

                else:
                    # follow orientation
                    s = self.v * dt
                    # print("orientation: ", self.orientation)
                    self.center = drive_straight(self.center, orientation=self.orientation, distance=s,
                                                 forward=control.forward)
                    self.from_center()

            # rotate according to control.psi
            # http://www.asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html section "curves"
            else:
                radius = control.psi / 180. * math.pi
                R = self._wheel_base / (math.sin(radius))
                # print(R / pixel_to_meter)
                if abs(R) < golf_turning_radius * pixel_to_meter:
                    if R > 0:
                        R = golf_turning_radius * pixel_to_meter
                    else:
                        R = -golf_turning_radius * pixel_to_meter
                # print(R / pixel_to_meter)
                omega = self.v / R
                if control.forward:
                    theta = omega * dt
                else:
                    theta = omega * dt * -1

                self.orientation += theta

                # center should only change when steering angle changed
                if control.psi != self.last_psi and control.psi != 0:
                    if control.psi > 0:  # left
                        self.turning_center = Point(
                            x=self.top_left.x - R * math.cos(radius),
                            y=self.top_left.y - R * math.sin(radius)
                        )
                    else:  # right
                        self.turning_center = Point(
                            x=self.top_right.x - R * math.cos(radius),
                            y=self.top_right.y - R * math.sin(radius)
                        )
                    self.last_psi = control.psi

                # print(self.turning_center.x, self.turning_center.y)

                # only rotate the middle point
                # works for left turn only
                self.center = (self.center - self.turning_center).rotate(theta) + self.turning_center
                self.from_center()

        l = [
            self.bottom_left.x, self.bottom_left.y,
            self.top_left.x, self.top_left.y,
            self.top_right.x, self.top_right.y,
            self.bottom_right.x, self.bottom_right.y
        ]
        return [int(e) for e in l]


class MyWindow(pyglet.window.Window):
    def __init__(self, height, width):
        self.control = CarControl()
        self.car = CarState()
        super().__init__(height=height, width=width)


def run():
    window = MyWindow(height=h + 2 * offset, width=w + 2 * offset)

    batch = pyglet.graphics.Batch()
    # draw 2 parked cars
    batch.add_indexed(4, pyglet.gl.GL_TRIANGLES, None,
                      [0, 1, 2, 0, 2, 3],
                      ('v2i', (offset, int(offset + 3 * car_h),
                               offset, int(offset + 4 * car_h),
                               offset + car_w, int(offset + 4 * car_h),
                               offset + car_w, int(offset + 3 * car_h)))
                      )
    batch.add_indexed(4, pyglet.gl.GL_TRIANGLES, None,
                      [0, 1, 2, 0, 2, 3],
                      ('v2i', (offset, offset,
                               offset, offset + car_h,
                               offset + car_w, offset + car_h,
                               offset + car_w, offset))

                      )
    # draw player car, start with a box first
    # todo: draw wheels + body, with a different color
    car_vertices = batch.add_indexed(4, pyglet.gl.GL_TRIANGLES, None,
                                     [0, 1, 2, 0, 2, 3],
                                     ('v2i', tuple(window.car.update_vertices(window.control)))
                                     )

    # draw a small rectangle represent wheel direction
    steering_vertices = batch.add_indexed(2, pyglet.gl.GL_LINES, None,
                                          [0, 1],
                                          ('v2i', (400, 600,
                                                   400, 540
                                                   )))

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.F:
            window.control.forward = True
        elif symbol == pyglet.window.key.R:
            window.control.forward = False
        elif symbol == pyglet.window.key.D:
            window.control.drive = not window.control.drive
        elif symbol == pyglet.window.key.UP:
            window.control.psi = 0
            steering_vertices.vertices = [400, 600, 400, 540]
            window.clear()
            batch.draw()

        # print("current control:", window.control)

    @window.event
    def on_text_motion(motion):
        if motion == pyglet.window.key.MOTION_LEFT:
            window.control.psi = min(52, window.control.psi + 10)
        elif motion == pyglet.window.key.MOTION_RIGHT:
            window.control.psi = max(-52, window.control.psi - 10)

        print("current control:", window.control)
        window.clear()
        r = window.control.psi / 180. * math.pi
        p1 = Point(0, 30).rotate(r) + Point(400, 570)
        p2 = Point(0, -30).rotate(r) + Point(400, 570)
        steering_vertices.vertices = [int(e) for e in [p1.x, p1.y, p2.x, p2.y]]
        batch.draw()

    def update(dt):
        '''
        This is where we update the controlled car positions
        '''
        # print("scheduled ", dt)
        # update car_vertices.vertices
        window.clear()  # important
        car_vertices.vertices = window.car.update_vertices(window.control, dt)
        batch.draw()
        # print("current control:", window.control)

    pyglet.clock.schedule_interval(update, 0.05)

    pyglet.app.run()


if __name__ == '__main__':
    run()
