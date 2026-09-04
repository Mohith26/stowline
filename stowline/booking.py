"""The booking decision itself, as a staged funnel.

A renter asks "can I put a 20 ft boat in this driveway from June to October".
Answering that honestly means four questions, cheapest first, because the last
one is exponential and the first one is three comparisons:

  1. dimensions   does the item physically fit the space and get through the
                  opening at all, ignoring everyone else
  2. area         is there enough square footage at the busiest instant of the
                  requested window (necessary, never sufficient)
  3. placement    is there an actual (x, y) for it that respects everyone
                  else's footprint and exit channel, without moving anybody
  4. repack       if not, would it have fit had every existing renter been
                  willing to shuffle

Stage 4 does not gate the booking. It is reported so the marketplace can see
how much inventory is lost purely to the "we cannot ask people to move their
stuff" constraint, which is a product decision, not an engineering one.
"""

from dataclasses import dataclass, field

from .geometry import Rect
from .placement import Reservation, solve, solve_exhaustive
from .timeline import peak_area

STAGES = ("dimensions", "area", "placement")


@dataclass
class Listing:
    id: str
    width: int
    depth: int
    door_width: int
    lat: float = 0.0
    lon: float = 0.0
    monthly_price: float = 0.0
    rating: float = 0.0
    covered: bool = False

    @property
    def area(self):
        return self.width * self.depth


@dataclass
class Decision:
    accepted: bool
    stage: str
    reason: str
    rect: object = None
    nodes: int = 0
    repack_would_fit: bool = None
    extras: dict = field(default_factory=dict)


def _fits_dimensions(listing, res):
    """Item has to fit the footprint and clear the opening.

    The door check uses the narrower of the two sides the item could be turned
    to, because you only have to get it through the opening once.
    """
    footprints = [(res.w, res.d)]
    if res.rotatable and res.w != res.d:
        footprints.append((res.d, res.w))
    ok_any = False
    for (w, d) in footprints:
        if w <= listing.width and d <= listing.depth and w <= listing.door_width:
            ok_any = True
    if not ok_any:
        return False, (
            "%dx%d does not fit a %dx%d space with a %d ft opening"
            % (res.w, res.d, listing.width, listing.depth, listing.door_width)
        )
    return True, "ok"


def quote(listing, existing, placement, request, check_repack=True, budget=400000):
    """Decide a single booking request.

    existing    list of Reservation already accepted on this listing
    placement   dict id -> Rect, where those items physically sit
    request     the Reservation being asked for
    """
    ok, why = _fits_dimensions(listing, request)
    if not ok:
        return Decision(False, "dimensions", why)

    window = [r for r in existing if r.start < request.end and request.start < r.end]
    needed = peak_area(window + [request])
    if needed > listing.area:
        return Decision(
            False,
            "area",
            "peak %d sq ft over the requested window exceeds the %d sq ft space"
            % (needed, listing.area),
            extras={"peak_area": needed, "space_area": listing.area},
        )

    fixed = [(r, placement[r.id]) for r in window if r.id in placement]
    feasible, full, nodes = solve(
        listing.width, listing.depth, window + [request], fixed=fixed, budget=budget
    )
    if feasible:
        return Decision(
            True,
            "placement",
            "placed",
            rect=full[request.id],
            nodes=nodes,
            extras={"peak_area": needed, "space_area": listing.area},
        )

    repack = None
    if check_repack:
        repack_feasible, _, _ = solve(
            listing.width, listing.depth, window + [request], fixed=[], budget=budget
        )
        repack = bool(repack_feasible)

    return Decision(
        False,
        "placement",
        "no legal position for %s without moving existing items" % request.id,
        nodes=nodes,
        repack_would_fit=repack,
        extras={"peak_area": needed, "space_area": listing.area},
    )


class ListingState:
    """Accepted reservations plus where each one physically sits."""

    def __init__(self, listing):
        self.listing = listing
        self.reservations = []
        self.placement = {}

    def try_book(self, request, **kw):
        decision = quote(
            self.listing, self.reservations, self.placement, request, **kw
        )
        if decision.accepted:
            self.reservations.append(request)
            self.placement[request.id] = decision.rect
        return decision

    def utilization_at(self, day):
        used = sum(
            r.w * r.d for r in self.reservations if r.start <= day < r.end
        )
        return used / float(self.listing.area)
