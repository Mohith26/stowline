"""Negative controls.

Two places in this codebase are written a specific way for a reason that is
easy to lose in a refactor, and "the tests pass" is not evidence that the
reason was real. So instead of asserting that the careful version is
necessary, this module reintroduces each mistake on purpose and measures what
it costs. If a control ever stops doing damage, either the test that guards it
went blind or the concern was imaginary.

Run: python -m stowline.controls
"""

import json
import math
import os
import random

from .generate import build_market
from .placement import Reservation
from .search import GridIndex, haversine_mi, query_radius_bruteforce
from .timeline import peak_area


class FlatDegreeIndex(GridIndex):
    """The longitude bug: convert the search radius to degrees with 69 miles
    per degree on both axes.

    That figure is correct for latitude everywhere and correct for longitude
    only at the equator. Longitude degrees shrink by cos(latitude), so at
    Utah's latitude a longitude degree is about 52.5 miles, and scanning
    radius/69 degrees of longitude covers only about 76% of the ground it
    needs to. The index silently returns fewer listings than exist.
    """

    def query_radius(self, lat, lon, radius_mi):
        deg = radius_mi / 69.0
        cells = int(math.ceil(deg / self.cell_deg))
        base_lat, base_lon = self._key(lat, lon)
        out = []
        examined = 0
        for i in range(base_lat - cells, base_lat + cells + 1):
            for j in range(base_lon - cells, base_lon + cells + 1):
                for listing in self.cells.get((i, j), ()):
                    examined += 1
                    dist = haversine_mi(lat, lon, listing.lat, listing.lon)
                    if dist <= radius_mi:
                        out.append((listing, dist))
        return out, examined


def peak_area_wrong_tie_order(reservations):
    """The interval bug: at a shared instant, apply arrivals before departures.

    With half open intervals `[start, end)`, an item ending on day 40 has
    already gone when an item starting on day 40 arrives. Sorting arrivals
    first makes two perfectly sequential stays look simultaneous, which
    inflates peak occupancy and rejects bookings that were always fine.
    """
    events = []
    for r in reservations:
        events.append((r.start, 0, r.w * r.d))
        events.append((r.end, 1, -(r.w * r.d)))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = 0
    peak = 0
    for _, _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def control_longitude(seed=31, n_listings=8000, queries=1200):
    listings, rng = build_market(seed, n_listings=n_listings)
    good = GridIndex(cell_deg=0.05)
    bad = FlatDegreeIndex(cell_deg=0.05)
    for listing in listings:
        good.insert(listing)
        bad.insert(listing)
    good_wrong = 0
    bad_wrong = 0
    dropped_total = 0
    truth_total = 0
    for _ in range(queries):
        lat = 40.3916 + rng.gauss(0, 0.4)
        lon = -111.8508 + rng.gauss(0, 0.4)
        radius = rng.choice([0.5, 2.0, 7.0, 25.0])
        truth = {l.id for l, _ in query_radius_bruteforce(listings, lat, lon, radius)}
        got_good = {l.id for l, _ in good.query_radius(lat, lon, radius)[0]}
        got_bad = {l.id for l, _ in bad.query_radius(lat, lon, radius)[0]}
        truth_total += len(truth)
        if got_good != truth:
            good_wrong += 1
        if got_bad != truth:
            bad_wrong += 1
            dropped_total += len(truth - got_bad)
    return {
        "queries": queries,
        "listings": n_listings,
        "correct_index_queries_wrong": good_wrong,
        "flat_degree_index_queries_wrong": bad_wrong,
        "flat_degree_listings_silently_dropped": dropped_total,
        "true_results_total": truth_total,
        "share_of_results_lost": round(dropped_total / float(max(1, truth_total)), 4),
        "longitude_miles_per_degree_at_lehi": round(69.0 * math.cos(math.radians(40.3916)), 2),
    }


def control_interval_ties(seed=32, trials=20000):
    rng = random.Random(seed)
    disagreements = 0
    inflated_total = 0
    for _ in range(trials):
        n = rng.randint(2, 5)
        items = []
        cursor = rng.randrange(0, 30)
        for i in range(n):
            length = rng.randint(5, 40)
            # deliberately chain some stays back to back, which is exactly the
            # pattern a listing sees when one renter replaces another
            if rng.random() < 0.6:
                start = cursor
            else:
                start = cursor + rng.randint(1, 10)
            items.append(Reservation("I%d" % i, rng.randint(2, 10), rng.randint(2, 10),
                                     start, start + length))
            cursor = start + length
        right = peak_area(items)
        wrong = peak_area_wrong_tie_order(items)
        if right != wrong:
            disagreements += 1
            inflated_total += wrong - right
    return {
        "trials": trials,
        "instances_where_tie_order_changes_the_answer": disagreements,
        "share": round(disagreements / float(trials), 4),
        "mean_square_feet_overstated_when_it_differs": round(
            inflated_total / float(max(1, disagreements)), 2
        ),
    }


def main():
    payload = {
        "longitude_scaling": control_longitude(),
        "interval_tie_order": control_interval_ties(),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/part_controls.json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
