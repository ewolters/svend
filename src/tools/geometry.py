"""
Geometry Tool - Coordinate Geometry and Geometric Calculations

Provides verified solutions for:
- Points, lines, circles, polygons
- Distances, areas, angles
- Intersections, transformations
- Triangle properties (centroid, circumcenter, incenter, etc.)
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, getcontext
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


# High precision for geometric calculations
getcontext().prec = 30

EPSILON = 1e-10


@dataclass
class Point:
    """2D point."""
    x: float
    y: float

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)

    def dot(self, other: 'Point') -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Point') -> float:
        """Cross product (z-component of 3D cross)."""
        return self.x * other.y - self.y * other.x

    def magnitude(self) -> float:
        """Vector magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> 'Point':
        """Unit vector."""
        mag = self.magnitude()
        if mag < EPSILON:
            return Point(0, 0)
        return self / mag

    def distance_to(self, other: 'Point') -> float:
        """Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def rotate(self, angle: float, origin: 'Point' = None) -> 'Point':
        """Rotate point around origin (radians)."""
        if origin is None:
            origin = Point(0, 0)
        translated = self - origin
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = Point(
            translated.x * cos_a - translated.y * sin_a,
            translated.x * sin_a + translated.y * cos_a
        )
        return rotated + origin

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Line:
    """Line defined by ax + by + c = 0."""
    a: float
    b: float
    c: float

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> 'Line':
        """Create line through two points."""
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = p2.x * p1.y - p1.x * p2.y
        return cls(a, b, c)

    @classmethod
    def from_slope_point(cls, slope: float, point: Point) -> 'Line':
        """Create line from slope and point."""
        # y - y1 = m(x - x1) => mx - y + (y1 - m*x1) = 0
        a = slope
        b = -1
        c = point.y - slope * point.x
        return cls(a, b, c)

    @classmethod
    def from_slope_intercept(cls, slope: float, y_intercept: float) -> 'Line':
        """Create line from y = mx + b form."""
        return cls(slope, -1, y_intercept)

    def slope(self) -> Optional[float]:
        """Get slope (None if vertical)."""
        if abs(self.b) < EPSILON:
            return None
        return -self.a / self.b

    def y_intercept(self) -> Optional[float]:
        """Get y-intercept (None if vertical)."""
        if abs(self.b) < EPSILON:
            return None
        return -self.c / self.b

    def x_intercept(self) -> Optional[float]:
        """Get x-intercept (None if horizontal)."""
        if abs(self.a) < EPSILON:
            return None
        return -self.c / self.a

    def distance_to_point(self, p: Point) -> float:
        """Perpendicular distance from point to line."""
        return abs(self.a * p.x + self.b * p.y + self.c) / math.sqrt(self.a ** 2 + self.b ** 2)

    def is_parallel_to(self, other: 'Line') -> bool:
        """Check if parallel to another line."""
        return abs(self.a * other.b - self.b * other.a) < EPSILON

    def is_perpendicular_to(self, other: 'Line') -> bool:
        """Check if perpendicular to another line."""
        return abs(self.a * other.a + self.b * other.b) < EPSILON

    def intersection(self, other: 'Line') -> Optional[Point]:
        """Find intersection point with another line."""
        det = self.a * other.b - other.a * self.b
        if abs(det) < EPSILON:
            return None  # Parallel
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return Point(x, y)

    def perpendicular_through(self, p: Point) -> 'Line':
        """Line perpendicular to this one through point p."""
        # Perpendicular has slope -b/a (negative reciprocal)
        return Line(self.b, -self.a, -self.b * p.x + self.a * p.y)

    def closest_point(self, p: Point) -> Point:
        """Closest point on line to given point."""
        perp = self.perpendicular_through(p)
        return self.intersection(perp)


@dataclass
class Circle:
    """Circle defined by center and radius."""
    center: Point
    radius: float

    @classmethod
    def from_three_points(cls, p1: Point, p2: Point, p3: Point) -> 'Circle':
        """Circumcircle through three points."""
        ax, ay = p1.x, p1.y
        bx, by = p2.x, p2.y
        cx, cy = p3.x, p3.y

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < EPSILON:
            raise ValueError("Points are collinear")

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

        center = Point(ux, uy)
        radius = center.distance_to(p1)
        return cls(center, radius)

    def contains_point(self, p: Point) -> bool:
        """Check if point is inside or on circle."""
        return self.center.distance_to(p) <= self.radius + EPSILON

    def point_position(self, p: Point) -> str:
        """Determine if point is inside, on, or outside circle."""
        dist = self.center.distance_to(p)
        if dist < self.radius - EPSILON:
            return "inside"
        elif dist > self.radius + EPSILON:
            return "outside"
        else:
            return "on"

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def circumference(self) -> float:
        return 2 * math.pi * self.radius

    def intersection_with_line(self, line: Line) -> List[Point]:
        """Find intersection points with a line."""
        # Distance from center to line
        dist = line.distance_to_point(self.center)

        if dist > self.radius + EPSILON:
            return []

        # Closest point on line to center
        closest = line.closest_point(self.center)

        if abs(dist - self.radius) < EPSILON:
            return [closest]  # Tangent

        # Two intersection points
        half_chord = math.sqrt(self.radius ** 2 - dist ** 2)
        direction = Point(-line.b, line.a).normalized()

        return [
            closest + direction * half_chord,
            closest - direction * half_chord
        ]

    def intersection_with_circle(self, other: 'Circle') -> List[Point]:
        """Find intersection points with another circle."""
        d = self.center.distance_to(other.center)

        # No intersection
        if d > self.radius + other.radius + EPSILON:
            return []
        if d < abs(self.radius - other.radius) - EPSILON:
            return []

        # One point (tangent)
        if abs(d - (self.radius + other.radius)) < EPSILON or abs(d - abs(self.radius - other.radius)) < EPSILON:
            direction = (other.center - self.center).normalized()
            return [self.center + direction * self.radius]

        # Two points
        a = (self.radius ** 2 - other.radius ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(self.radius ** 2 - a ** 2)

        direction = (other.center - self.center).normalized()
        mid = self.center + direction * a
        perp = Point(-direction.y, direction.x)

        return [
            mid + perp * h,
            mid - perp * h
        ]


class Triangle:
    """Triangle with various properties."""

    def __init__(self, a: Point, b: Point, c: Point):
        self.a = a
        self.b = b
        self.c = c

    @property
    def vertices(self) -> Tuple[Point, Point, Point]:
        return (self.a, self.b, self.c)

    def side_lengths(self) -> Tuple[float, float, float]:
        """Lengths of sides (opposite to vertices a, b, c)."""
        return (
            self.b.distance_to(self.c),  # side a
            self.c.distance_to(self.a),  # side b
            self.a.distance_to(self.b),  # side c
        )

    def perimeter(self) -> float:
        return sum(self.side_lengths())

    def area(self) -> float:
        """Area via cross product."""
        ab = self.b - self.a
        ac = self.c - self.a
        return abs(ab.cross(ac)) / 2

    def area_herons(self) -> float:
        """Area via Heron's formula."""
        a, b, c = self.side_lengths()
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))

    def centroid(self) -> Point:
        """Centroid (intersection of medians)."""
        return Point(
            (self.a.x + self.b.x + self.c.x) / 3,
            (self.a.y + self.b.y + self.c.y) / 3
        )

    def circumcenter(self) -> Point:
        """Circumcenter (equidistant from vertices)."""
        circle = Circle.from_three_points(self.a, self.b, self.c)
        return circle.center

    def circumradius(self) -> float:
        """Radius of circumscribed circle."""
        circle = Circle.from_three_points(self.a, self.b, self.c)
        return circle.radius

    def incenter(self) -> Point:
        """Incenter (equidistant from sides)."""
        a, b, c = self.side_lengths()
        total = a + b + c
        return Point(
            (a * self.a.x + b * self.b.x + c * self.c.x) / total,
            (a * self.a.y + b * self.b.y + c * self.c.y) / total
        )

    def inradius(self) -> float:
        """Radius of inscribed circle."""
        return 2 * self.area() / self.perimeter()

    def orthocenter(self) -> Point:
        """Orthocenter (intersection of altitudes)."""
        # Altitude from A perpendicular to BC
        bc = Line.from_points(self.b, self.c)
        alt_a = bc.perpendicular_through(self.a)

        # Altitude from B perpendicular to AC
        ac = Line.from_points(self.a, self.c)
        alt_b = ac.perpendicular_through(self.b)

        return alt_a.intersection(alt_b)

    def angles(self) -> Tuple[float, float, float]:
        """Interior angles at vertices A, B, C (in radians)."""
        a, b, c = self.side_lengths()

        # Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
        angle_a = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
        angle_b = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
        angle_c = math.acos((a**2 + b**2 - c**2) / (2 * a * b))

        return (angle_a, angle_b, angle_c)

    def classify(self) -> Dict[str, str]:
        """Classify triangle by sides and angles."""
        a, b, c = self.side_lengths()
        angles = self.angles()

        # By sides
        if abs(a - b) < EPSILON and abs(b - c) < EPSILON:
            by_sides = "equilateral"
        elif abs(a - b) < EPSILON or abs(b - c) < EPSILON or abs(a - c) < EPSILON:
            by_sides = "isosceles"
        else:
            by_sides = "scalene"

        # By angles
        max_angle = max(angles)
        if abs(max_angle - math.pi / 2) < EPSILON:
            by_angles = "right"
        elif max_angle > math.pi / 2:
            by_angles = "obtuse"
        else:
            by_angles = "acute"

        return {"by_sides": by_sides, "by_angles": by_angles}


class Polygon:
    """General polygon."""

    def __init__(self, vertices: List[Point]):
        if len(vertices) < 3:
            raise ValueError("Polygon needs at least 3 vertices")
        self.vertices = vertices
        self.n = len(vertices)

    def area(self) -> float:
        """Area via shoelace formula."""
        total = 0
        for i in range(self.n):
            j = (i + 1) % self.n
            total += self.vertices[i].x * self.vertices[j].y
            total -= self.vertices[j].x * self.vertices[i].y
        return abs(total) / 2

    def perimeter(self) -> float:
        """Sum of side lengths."""
        total = 0
        for i in range(self.n):
            j = (i + 1) % self.n
            total += self.vertices[i].distance_to(self.vertices[j])
        return total

    def centroid(self) -> Point:
        """Centroid of polygon."""
        cx, cy = 0, 0
        area = self.area()

        for i in range(self.n):
            j = (i + 1) % self.n
            factor = self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y
            cx += (self.vertices[i].x + self.vertices[j].x) * factor
            cy += (self.vertices[i].y + self.vertices[j].y) * factor

        return Point(cx / (6 * area), cy / (6 * area))

    def is_convex(self) -> bool:
        """Check if polygon is convex."""
        if self.n < 3:
            return False

        sign = None
        for i in range(self.n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % self.n]
            p3 = self.vertices[(i + 2) % self.n]

            cross = (p2 - p1).cross(p3 - p2)

            if abs(cross) > EPSILON:
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False

        return True

    def contains_point(self, p: Point) -> bool:
        """Check if point is inside polygon (ray casting)."""
        inside = False
        j = self.n - 1

        for i in range(self.n):
            vi, vj = self.vertices[i], self.vertices[j]

            if ((vi.y > p.y) != (vj.y > p.y)) and \
               (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x):
                inside = not inside

            j = i

        return inside


# Tool implementation

def geometry_tool(
    operation: str,
    points: Optional[List[List[float]]] = None,
    line1: Optional[Dict] = None,
    line2: Optional[Dict] = None,
    circle: Optional[Dict] = None,
    circle2: Optional[Dict] = None,
    angle: Optional[float] = None,
    angle_unit: str = "degrees",
) -> ToolResult:
    """Execute geometry operation."""
    try:
        # Convert points
        pts = [Point(p[0], p[1]) for p in points] if points else []

        if operation == "distance":
            if len(pts) < 2:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 2 points")
            d = pts[0].distance_to(pts[1])
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Distance from {pts[0].to_tuple()} to {pts[1].to_tuple()} = {d:.10g}",
                metadata={"distance": d}
            )

        elif operation == "midpoint":
            if len(pts) < 2:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 2 points")
            mid = (pts[0] + pts[1]) / 2
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Midpoint of {pts[0].to_tuple()} and {pts[1].to_tuple()} = {mid.to_tuple()}",
                metadata={"midpoint": mid.to_tuple()}
            )

        elif operation == "line_from_points":
            if len(pts) < 2:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 2 points")
            line = Line.from_points(pts[0], pts[1])
            slope = line.slope()
            y_int = line.y_intercept()
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Line: {line.a:.6g}x + {line.b:.6g}y + {line.c:.6g} = 0, slope = {slope if slope is not None else 'undefined'}, y-intercept = {y_int if y_int is not None else 'none'}",
                metadata={"a": line.a, "b": line.b, "c": line.c, "slope": slope, "y_intercept": y_int}
            )

        elif operation == "line_intersection":
            if not line1 or not line2:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need two lines (a, b, c coefficients)")
            l1 = Line(line1["a"], line1["b"], line1["c"])
            l2 = Line(line2["a"], line2["b"], line2["c"])
            pt = l1.intersection(l2)
            if pt is None:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="Lines are parallel (no intersection)",
                    metadata={"parallel": True, "intersection": None}
                )
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Intersection point: {pt.to_tuple()}",
                metadata={"intersection": pt.to_tuple()}
            )

        elif operation == "point_to_line_distance":
            if len(pts) < 1 or not line1:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need point and line")
            l = Line(line1["a"], line1["b"], line1["c"])
            d = l.distance_to_point(pts[0])
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Distance from {pts[0].to_tuple()} to line = {d:.10g}",
                metadata={"distance": d}
            )

        elif operation == "circle_from_points":
            if len(pts) < 3:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 3 points")
            try:
                c = Circle.from_three_points(pts[0], pts[1], pts[2])
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Circle: center = {c.center.to_tuple()}, radius = {c.radius:.10g}",
                    metadata={"center": c.center.to_tuple(), "radius": c.radius}
                )
            except ValueError as e:
                return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))

        elif operation == "circle_line_intersection":
            if not circle or not line1:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need circle and line")
            c = Circle(Point(circle["center"][0], circle["center"][1]), circle["radius"])
            l = Line(line1["a"], line1["b"], line1["c"])
            intersections = c.intersection_with_line(l)
            pts_out = [p.to_tuple() for p in intersections]
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Intersections ({len(pts_out)}): {pts_out}",
                metadata={"count": len(pts_out), "points": pts_out}
            )

        elif operation == "triangle":
            if len(pts) < 3:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 3 points")
            tri = Triangle(pts[0], pts[1], pts[2])
            sides = tri.side_lengths()
            angles_rad = tri.angles()
            angles_deg = tuple(math.degrees(a) for a in angles_rad)
            classification = tri.classify()

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Triangle:\n  Sides: a={sides[0]:.6g}, b={sides[1]:.6g}, c={sides[2]:.6g}\n  Angles: A={angles_deg[0]:.4g}°, B={angles_deg[1]:.4g}°, C={angles_deg[2]:.4g}°\n  Area: {tri.area():.10g}\n  Perimeter: {tri.perimeter():.10g}\n  Type: {classification['by_sides']}, {classification['by_angles']}\n  Centroid: {tri.centroid().to_tuple()}\n  Incenter: {tri.incenter().to_tuple()}\n  Circumcenter: {tri.circumcenter().to_tuple()}",
                metadata={
                    "sides": sides,
                    "angles_degrees": angles_deg,
                    "angles_radians": angles_rad,
                    "area": tri.area(),
                    "perimeter": tri.perimeter(),
                    "classification": classification,
                    "centroid": tri.centroid().to_tuple(),
                    "incenter": tri.incenter().to_tuple(),
                    "circumcenter": tri.circumcenter().to_tuple(),
                    "inradius": tri.inradius(),
                    "circumradius": tri.circumradius(),
                }
            )

        elif operation == "polygon_area":
            if len(pts) < 3:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need at least 3 points")
            poly = Polygon(pts)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Polygon area: {poly.area():.10g}, perimeter: {poly.perimeter():.10g}, convex: {poly.is_convex()}",
                metadata={"area": poly.area(), "perimeter": poly.perimeter(), "is_convex": poly.is_convex()}
            )

        elif operation == "rotate_point":
            if len(pts) < 1 or angle is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need point and angle")
            origin = pts[1] if len(pts) > 1 else Point(0, 0)
            angle_rad = math.radians(angle) if angle_unit == "degrees" else angle
            rotated = pts[0].rotate(angle_rad, origin)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Rotated point: {rotated.to_tuple()}",
                metadata={"rotated": rotated.to_tuple()}
            )

        elif operation == "angle_between_vectors":
            if len(pts) < 2:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need 2 vectors (as points)")
            v1, v2 = pts[0], pts[1]
            dot = v1.dot(v2)
            mags = v1.magnitude() * v2.magnitude()
            if mags < EPSILON:
                return ToolResult(status=ToolStatus.ERROR, output=None, error="Zero-length vector")
            angle_rad = math.acos(max(-1, min(1, dot / mags)))
            angle_deg = math.degrees(angle_rad)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Angle between vectors: {angle_deg:.6g}° ({angle_rad:.10g} radians)",
                metadata={"degrees": angle_deg, "radians": angle_rad}
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: distance, midpoint, line_from_points, line_intersection, point_to_line_distance, circle_from_points, circle_line_intersection, triangle, polygon_area, rotate_point, angle_between_vectors"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_geometry_tool() -> Tool:
    """Create geometry tool."""
    return Tool(
        name="geometry",
        description="Coordinate geometry: distances, lines, circles, triangles (with centroid, incenter, circumcenter, orthocenter), polygons, intersections, rotations.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: distance, midpoint, line_from_points, line_intersection, point_to_line_distance, circle_from_points, circle_line_intersection, triangle, polygon_area, rotate_point, angle_between_vectors",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="points",
                description="List of points as [[x1, y1], [x2, y2], ...]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="line1",
                description="First line as {a, b, c} for ax + by + c = 0",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="line2",
                description="Second line as {a, b, c}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="circle",
                description="Circle as {center: [x, y], radius: r}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="angle",
                description="Angle for rotation",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="angle_unit",
                description="Unit for angle: degrees (default) or radians",
                type="string",
                required=False,
                enum=["degrees", "radians"],
            ),
        ],
        execute_fn=geometry_tool,
        timeout_ms=10000,
    )


def register_geometry_tools(registry: ToolRegistry) -> None:
    """Register geometry tools."""
    registry.register(create_geometry_tool())
