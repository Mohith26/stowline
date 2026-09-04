"""Axis-aligned rectangle geometry for physical storage spaces.

Coordinate convention used everywhere in this package:

    y = 0 is the access edge (the garage door, the mouth of the driveway,
    the gate of the lot). y grows away from it, into the space. x runs
    left-to-right across the opening.

Everything is in feet, kept as integers. Storage listings are advertised in
whole feet (10x10, 10x20, "24 ft RV space"), so integer feet is the natural
unit and it keeps the exhaustive oracle in placement.py finite.
"""

from collections import namedtuple

# Lower-left corner at (x, y), extent w across the opening, extent d into the space.
Rect = namedtuple("Rect", "x y w d")


def area(r):
    return r.w * r.d


def overlaps(a, b):
    """True if two rectangles share interior area. Touching edges is allowed."""
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.d and b.y < a.y + a.d


def contains(outer_w, outer_d, r):
    """True if r sits entirely inside a space of outer_w x outer_d."""
    return r.x >= 0 and r.y >= 0 and r.x + r.w <= outer_w and r.y + r.d <= outer_d


def corridor(r):
    """The strip an item has to traverse to reach the access edge.

    An item sitting at depth y has to come straight out toward y = 0 through a
    channel exactly as wide as it is. Anything sitting in that channel is in
    the way. This is the same "blocked-in car" idea a valet lot has, and it is
    what makes a storage listing different from an abstract capacity counter:
    two items can be non-overlapping and still be an illegal arrangement.

    Returns None when the item already touches the access edge (nothing to
    traverse).
    """
    if r.y <= 0:
        return None
    return Rect(r.x, 0, r.w, r.y)


def blocks(blocker, mover):
    """True if `blocker` sits in the channel `mover` needs to get out."""
    c = corridor(mover)
    if c is None:
        return False
    return overlaps(blocker, c)


def orientations(w, d, rotatable):
    """Candidate footprints for an item.

    Boxes and pallets can be turned; a 24 ft travel trailer in a 12 ft wide
    driveway cannot, so rotation is a per-item property rather than a global
    assumption.
    """
    if rotatable and w != d:
        return [(w, d), (d, w)]
    return [(w, d)]
