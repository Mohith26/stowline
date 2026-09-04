"""Finding candidate listings near a renter.

A renter searching "storage near 84043" needs the nearby listings ranked, and
the expensive per-listing work (the placement search in placement.py) should
only ever run on listings that survived the cheap geographic filter. So this
module is a uniform grid index over latitude/longitude, sized so that a radius
query touches a small constant number of cells.

Grid, not a k-d tree or an R-tree, on purpose: listings churn constantly as
hosts add and pause spaces, and a grid takes O(1) insert and delete with no
rebalancing. The cost is sensitivity to cell size, which experiments.py
measures rather than guesses at.
"""

import math

EARTH_RADIUS_MI = 3958.8


def haversine_mi(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


class GridIndex:
    def __init__(self, cell_deg=0.05):
        self.cell_deg = cell_deg
        self.cells = {}
        self.count = 0

    def _key(self, lat, lon):
        return (int(math.floor(lat / self.cell_deg)), int(math.floor(lon / self.cell_deg)))

    def insert(self, listing):
        self.cells.setdefault(self._key(listing.lat, listing.lon), []).append(listing)
        self.count += 1

    def remove(self, listing):
        key = self._key(listing.lat, listing.lon)
        bucket = self.cells.get(key, [])
        for i, item in enumerate(bucket):
            if item.id == listing.id:
                bucket.pop(i)
                self.count -= 1
                return True
        return False

    def query_radius(self, lat, lon, radius_mi):
        """Listings within radius_mi, plus the number of candidates examined.

        The cell span is computed from the true degrees-per-mile at this
        latitude in each axis separately. Longitude degrees shrink by cos(lat),
        so a fixed square grid in degrees is a tall thin rectangle in miles at
        Utah's latitude, and using the latitude conversion for both axes would
        silently under-scan the longitude direction and drop real results.
        """
        lat_deg = radius_mi / 69.0
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        lon_deg = radius_mi / (69.0 * cos_lat)
        lat_cells = int(math.ceil(lat_deg / self.cell_deg))
        lon_cells = int(math.ceil(lon_deg / self.cell_deg))
        base_lat, base_lon = self._key(lat, lon)
        out = []
        examined = 0
        for i in range(base_lat - lat_cells, base_lat + lat_cells + 1):
            for j in range(base_lon - lon_cells, base_lon + lon_cells + 1):
                for listing in self.cells.get((i, j), ()):
                    examined += 1
                    dist = haversine_mi(lat, lon, listing.lat, listing.lon)
                    if dist <= radius_mi:
                        out.append((listing, dist))
        return out, examined


def query_radius_bruteforce(listings, lat, lon, radius_mi):
    out = []
    for listing in listings:
        dist = haversine_mi(lat, lon, listing.lat, listing.lon)
        if dist <= radius_mi:
            out.append((listing, dist))
    return out


def rank(results, item_area, weights=(0.45, 0.35, 0.20)):
    """Order candidates for a renter.

    Normalizing distance and price against the max in the result set rather
    than a fixed constant keeps the weights meaningful whether the renter is in
    a dense metro or searching a 40 mile radius of nothing.
    """
    if not results:
        return []
    w_dist, w_price, w_rating = weights
    max_dist = max(d for _, d in results) or 1e-6
    max_price = max(l.monthly_price for l, _ in results) or 1e-6
    scored = []
    for listing, dist in results:
        dist_score = 1.0 - dist / max_dist
        price_score = 1.0 - listing.monthly_price / max_price
        rating_score = listing.rating / 5.0
        score = w_dist * dist_score + w_price * price_score + w_rating * rating_score
        scored.append((listing, dist, score))
    scored.sort(key=lambda t: (-t[2], t[0].id))
    return scored
