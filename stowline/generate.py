"""Seeded synthetic market, so every number in results/results.json is
reproducible from a seed rather than from whatever was in memory that day.

Sizes are drawn from the shapes real peer-to-peer storage listings actually
come in: single-car garages, two-car garages, driveways, and open lots, plus
the item mix those spaces get asked to hold (sedans, trucks, boats on
trailers, travel trailers, and pallets of household boxes).
"""

import random

from .booking import Listing
from .placement import Reservation

# (label, width_ft, depth_ft, door_width_ft)
SPACE_TYPES = [
    ("single_garage", 12, 22, 9),
    ("double_garage", 20, 22, 16),
    ("tandem_garage", 12, 40, 9),
    ("driveway", 10, 30, 10),
    ("wide_driveway", 20, 30, 20),
    ("open_lot", 30, 40, 30),
]

# (label, width_ft, depth_ft, rotatable)
ITEM_TYPES = [
    ("sedan", 6, 15, False),
    ("suv", 7, 17, False),
    ("pickup", 7, 20, False),
    ("boat_trailer", 8, 22, False),
    ("travel_trailer", 8, 26, False),
    ("motorcycle", 3, 7, False),
    ("box_pallet_small", 5, 5, True),
    ("box_pallet_large", 10, 10, True),
    ("furniture_block", 8, 12, True),
]


def make_listing(rng, listing_id, center=(40.3916, -111.8508), spread=0.25):
    """Default centre is Lehi, Utah, since that is a real dense market for this
    kind of listing and it keeps the latitude realistic for the longitude
    scaling in search.py."""
    label, w, d, door = rng.choice(SPACE_TYPES)
    lat = center[0] + rng.gauss(0, spread)
    lon = center[1] + rng.gauss(0, spread)
    return Listing(
        id="L%04d" % listing_id,
        width=w,
        depth=d,
        door_width=door,
        lat=lat,
        lon=lon,
        monthly_price=0.0,
        rating=round(min(5.0, max(3.0, rng.gauss(4.5, 0.4))), 2),
        covered=label.endswith("garage"),
    )


def make_request(rng, req_id, horizon_days=365):
    label, w, d, rot = rng.choice(ITEM_TYPES)
    start = rng.randrange(0, horizon_days - 30)
    length = rng.choice([30, 60, 90, 120, 180, 240, 365 - start if start < 30 else 180])
    length = max(20, min(length, horizon_days - start))
    return Reservation(
        id="R%05d" % req_id,
        w=w,
        d=d,
        start=start,
        end=start + length,
        rotatable=rot,
    ), label


def true_price(listing, rng, center=(40.3916, -111.8508)):
    """Ground-truth monthly price for the synthetic market.

    Built as a smooth spatial surface (price falls with distance from the
    centre) times a size power law times a covered-space premium times noise,
    so the pricing model in pricing.py has something real to be scored
    against and cannot trivially recover it.
    """
    from .search import haversine_mi

    dist = haversine_mi(center[0], center[1], listing.lat, listing.lon)
    base_ppsf = 0.95 * (1.0 / (1.0 + dist / 14.0))
    size_factor = (listing.area / 200.0) ** -0.15
    covered_premium = 1.25 if listing.covered else 1.0
    noise = rng.lognormvariate(0.0, 0.12)
    return round(base_ppsf * listing.area * size_factor * covered_premium * noise, 2)


def build_market(seed, n_listings=400):
    rng = random.Random(seed)
    listings = []
    for i in range(n_listings):
        listing = make_listing(rng, i)
        listing.monthly_price = true_price(listing, rng)
        listings.append(listing)
    return listings, rng


def random_instance(rng, space_w, space_d, n_items, horizon=200, max_side=None):
    """A small packing instance for oracle comparison.

    Item sizes are drawn small enough relative to the space that a meaningful
    fraction of instances are feasible; an instance set that is 99% infeasible
    tells you nothing about whether the fast solver misses solutions.
    """
    items = []
    for i in range(n_items):
        w = rng.randint(2, max_side or max(2, space_w // 2))
        d = rng.randint(2, max_side or max(2, space_d // 2))
        start = rng.randrange(0, horizon - 20)
        end = start + rng.randrange(10, horizon - start + 1)
        items.append(
            Reservation(
                id="I%d" % i,
                w=w,
                d=d,
                start=start,
                end=end,
                rotatable=rng.random() < 0.4,
            )
        )
    return items
