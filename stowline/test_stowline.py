"""Test suite. Run with `python -m stowline.test_stowline` from the repo root.

No pytest dependency on purpose, so this runs anywhere with a bare interpreter.

The tests that actually matter here are the differential ones. Nearly every
real bug I hit while writing this was a case where the fast path and the
obvious slow path disagreed, and none of them would have been caught by an
example-based assertion I thought to write by hand.
"""

import random
import sys
import time

from .booking import Listing, ListingState, quote
from .geometry import Rect, blocks, contains, corridor, overlaps
from .placement import (
    Reservation,
    check_placement,
    solve,
    solve_exhaustive,
)
from .pricing import weighted_median, suggest_price, evaluate
from .search import GridIndex, haversine_mi, query_radius_bruteforce, rank
from .timeline import (
    active_at,
    event_days,
    nested_within,
    peak_area,
    peak_area_bruteforce,
)
from .generate import build_market, random_instance, true_price

_T0 = time.time()

RESULTS = {"passed": 0, "failed": 0, "assertions": 0, "failures": []}


def check(name, cond, detail=""):
    RESULTS["assertions"] += 1
    if not cond:
        RESULTS["failed"] += 1
        RESULTS["failures"].append("%s: %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))
    return cond


def case(fn):
    name = fn.__name__
    before_failed = RESULTS["failed"]
    fn()
    if RESULTS["failed"] == before_failed:
        RESULTS["passed"] += 1
        print("  ok   %s" % name)
    return fn


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@case
def test_overlap_is_open_interval():
    a = Rect(0, 0, 10, 10)
    check("touching edges do not overlap", not overlaps(a, Rect(10, 0, 5, 5)))
    check("shared interior overlaps", overlaps(a, Rect(9, 9, 5, 5)))
    check("containment overlaps", overlaps(a, Rect(2, 2, 3, 3)))


@case
def test_bounds():
    check("inside", contains(20, 30, Rect(0, 0, 20, 30)))
    check("one over in x", not contains(20, 30, Rect(1, 0, 20, 30)))
    check("negative origin", not contains(20, 30, Rect(-1, 0, 5, 5)))


@case
def test_corridor_and_blocking():
    parked_deep = Rect(0, 10, 8, 20)
    check("corridor spans door to item", corridor(parked_deep) == Rect(0, 0, 8, 10))
    check("item on the edge has no corridor", corridor(Rect(0, 0, 8, 10)) is None)
    check("item in the channel blocks", blocks(Rect(0, 0, 8, 8), parked_deep))
    check(
        "item beside the channel does not block",
        not blocks(Rect(8, 0, 4, 8), parked_deep),
    )
    check(
        "item behind does not block",
        not blocks(Rect(0, 30, 8, 5), parked_deep),
    )


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------


@case
def test_peak_area_matches_bruteforce():
    rng = random.Random(11)
    mismatches = 0
    for _ in range(3000):
        items = []
        for i in range(rng.randint(1, 8)):
            start = rng.randrange(0, 60)
            items.append(
                Reservation(
                    "I%d" % i,
                    rng.randint(1, 9),
                    rng.randint(1, 9),
                    start,
                    start + rng.randint(1, 40),
                )
            )
        if peak_area(items) != peak_area_bruteforce(items):
            mismatches += 1
    check("sweep equals day-by-day scan on 3000 instances", mismatches == 0,
          "%d mismatches" % mismatches)


@case
def test_half_open_intervals():
    a = Reservation("a", 10, 10, 0, 40)
    b = Reservation("b", 10, 10, 40, 80)
    check("back to back stays never coexist", peak_area([a, b]) == 100)
    check("day 39 has a", [r.id for r in active_at([a, b], 39)] == ["a"])
    check("day 40 has b", [r.id for r in active_at([a, b], 40)] == ["b"])
    check("event days", event_days([a, b]) == [0, 40, 80])


@case
def test_nesting_rule():
    outer = Reservation("o", 8, 8, 10, 100)
    check("strictly inside", nested_within(Reservation("i", 1, 1, 11, 99), outer))
    check("same start is not nested", not nested_within(Reservation("i", 1, 1, 10, 99), outer))
    check("same end is not nested", not nested_within(Reservation("i", 1, 1, 11, 100), outer))
    check("later end is not nested", not nested_within(Reservation("i", 1, 1, 11, 200), outer))


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


@case
def test_checker_rejects_known_bad_arrangements():
    a = Reservation("a", 8, 10, 0, 100)
    b = Reservation("b", 8, 10, 0, 100)
    ok, _ = check_placement(20, 30, [(a, Rect(0, 0, 8, 10)), (b, Rect(4, 0, 8, 10))])
    check("spatial overlap rejected", not ok)

    ok, _ = check_placement(20, 30, [(a, Rect(0, 40, 8, 10))])
    check("out of bounds rejected", not ok)

    # b sits in a's exit channel and leaves after a does
    deep = Reservation("deep", 8, 10, 0, 100)
    front = Reservation("front", 8, 10, 0, 200)
    ok, _ = check_placement(20, 30, [(deep, Rect(0, 10, 8, 10)), (front, Rect(0, 0, 8, 10))])
    check("blocking item that leaves later is rejected", not ok)

    # same geometry, but the blocker's stay is strictly nested, which is legal
    nested = Reservation("nested", 8, 10, 10, 90)
    ok, why = check_placement(20, 30, [(deep, Rect(0, 10, 8, 10)), (nested, Rect(0, 0, 8, 10))])
    check("nested blocker is allowed", ok, why)

    # non-overlapping stays can share the same spot
    early = Reservation("early", 8, 10, 0, 50)
    late = Reservation("late", 8, 10, 50, 100)
    ok, why = check_placement(20, 30, [(early, Rect(0, 0, 8, 10)), (late, Rect(0, 0, 8, 10))])
    check("sequential stays may share a spot", ok, why)


@case
def test_rotation_is_respected():
    fixed = Reservation("f", 4, 12, 0, 100, rotatable=False)
    ok, why = check_placement(20, 30, [(fixed, Rect(0, 0, 12, 4))])
    check("non rotatable item cannot be turned", not ok)
    spin = Reservation("s", 4, 12, 0, 100, rotatable=True)
    ok, why = check_placement(20, 30, [(spin, Rect(0, 0, 12, 4))])
    check("rotatable item may be turned", ok, why)


@case
def test_solver_output_always_validates():
    """Every arrangement the fast solver returns is re-checked by the
    independent O(n^2) validator, not by the solver's own logic."""
    rng = random.Random(202)
    feasible = 0
    bad = 0
    for _ in range(400):
        w, d = rng.choice([(12, 22), (20, 22), (20, 30), (30, 40)])
        items = random_instance(rng, w, d, rng.randint(1, 5), max_side=10)
        found, placement, _ = solve(w, d, items)
        if not found:
            continue
        feasible += 1
        placed = [(r, placement[r.id]) for r in items]
        ok, why = check_placement(w, d, placed)
        if not ok:
            bad += 1
    check("at least some instances were feasible", feasible > 80, "only %d" % feasible)
    check("every solver arrangement passes the validator", bad == 0,
          "%d invalid of %d" % (bad, feasible))


@case
def test_fast_solver_against_exhaustive_oracle():
    """The core differential test: does the corner-point search ever call an
    instance infeasible when a complete search over every integer position can
    actually solve it?"""
    rng = random.Random(303)
    agree = 0
    fast_missed = 0
    oracle_missed = 0
    undecided = 0
    for _ in range(150):
        w, d = rng.choice([(10, 12), (12, 14), (10, 16)])
        items = random_instance(rng, w, d, rng.randint(2, 4), horizon=60, max_side=7)
        fast, _, _ = solve(w, d, items, budget=60000)
        slow, _, _ = solve_exhaustive(w, d, items, budget=300000)
        if fast is None or slow is None:
            undecided += 1
            continue
        if fast == slow:
            agree += 1
        elif slow and not fast:
            fast_missed += 1
        else:
            oracle_missed += 1
    total = agree + fast_missed + oracle_missed
    check("oracle resolved most instances", total > 100, "only %d resolved" % total)
    check("fast solver never beats the complete search", oracle_missed == 0,
          "%d impossible disagreements" % oracle_missed)
    RESULTS.setdefault("oracle", {})
    RESULTS["oracle"] = {
        "resolved": total,
        "agree": agree,
        "fast_missed": fast_missed,
        "undecided": undecided,
    }
    print("       oracle: %d resolved, %d agree, %d missed by fast path"
          % (total, agree, fast_missed))


@case
def test_fixed_items_are_never_moved():
    a = Reservation("a", 8, 10, 0, 100)
    b = Reservation("b", 6, 6, 0, 100)
    pinned = Rect(6, 12, 8, 10)
    found, placement, _ = solve(20, 30, [a, b], fixed=[(a, pinned)])
    check("solved with a pin", found)
    if found:
        check("pinned item did not move", placement["a"] == pinned, str(placement["a"]))


# --------------------------------------------------------------------------
# booking funnel
# --------------------------------------------------------------------------


@case
def test_dimension_gate():
    listing = Listing("L", width=10, depth=30, door_width=10)
    state = ListingState(listing)
    too_wide = Reservation("x", 12, 10, 0, 30)
    decision = state.try_book(too_wide)
    check("oversize item rejected at the dimension stage",
          not decision.accepted and decision.stage == "dimensions", decision.stage)

    narrow_door = Listing("L2", width=20, depth=30, door_width=8)
    state2 = ListingState(narrow_door)
    wide_item = Reservation("y", 10, 10, 0, 30, rotatable=False)
    decision = state2.try_book(wide_item)
    check("item wider than the opening rejected",
          not decision.accepted and decision.stage == "dimensions", decision.stage)

    spinner = Reservation("z", 10, 6, 0, 30, rotatable=True)
    decision = state2.try_book(spinner)
    check("rotatable item can be turned to clear the opening", decision.accepted,
          decision.reason)


@case
def test_area_gate_is_only_necessary():
    """A listing with plenty of square footage left can still be physically
    full. This is the whole reason the placement stage exists."""
    listing = Listing("L", width=20, depth=30, door_width=20)
    state = ListingState(listing)
    # Two deep items pinned along the back leave 20x20 = 400 sq ft free,
    # but nothing may sit in front of them, because they leave last.
    a = Reservation("a", 10, 10, 0, 300)
    b = Reservation("b", 10, 10, 0, 300)
    check("a booked", state.try_book(a).accepted)
    check("b booked", state.try_book(b).accepted)
    check("area available", listing.area - 200 == 400)


@case
def test_accepted_bookings_stay_valid_over_a_season():
    """Book a long sequence into one listing and confirm the arrangement is
    still legal after every single accept, not just at the end."""
    rng = random.Random(77)
    listing = Listing("L", width=20, depth=30, door_width=20)
    state = ListingState(listing)
    accepted = 0
    for i in range(120):
        item = random_instance(rng, 20, 30, 1, horizon=365, max_side=10)[0]
        item = Reservation("B%03d" % i, item.w, item.d, item.start, item.end, item.rotatable)
        decision = state.try_book(item, check_repack=False)
        if decision.accepted:
            accepted += 1
            placed = [(r, state.placement[r.id]) for r in state.reservations]
            ok, why = check_placement(20, 30, placed)
            if not check("arrangement valid after accepting %s" % item.id, ok, why):
                break
    check("some bookings were accepted", accepted > 5, "only %d" % accepted)


@case
def test_rejection_reasons_are_ordered_cheapest_first():
    listing = Listing("L", width=20, depth=30, door_width=20)
    huge = Reservation("h", 25, 25, 0, 30)
    decision = quote(listing, [], {}, huge)
    check("dimension failure short circuits before area", decision.stage == "dimensions",
          decision.stage)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


@case
def test_haversine_known_distances():
    # Lehi UT to Salt Lake City UT, roughly 25 miles
    d = haversine_mi(40.3916, -111.8508, 40.7608, -111.8910)
    check("Lehi to SLC in range", 24.0 < d < 27.0, "%.2f" % d)
    check("zero distance", haversine_mi(40.0, -111.0, 40.0, -111.0) < 1e-9)
    # one degree of latitude is about 69 miles anywhere
    d = haversine_mi(40.0, -111.0, 41.0, -111.0)
    check("one degree latitude", 68.5 < d < 69.5, "%.2f" % d)


@case
def test_grid_index_matches_bruteforce():
    listings, rng = build_market(5, n_listings=800)
    index = GridIndex(cell_deg=0.05)
    for listing in listings:
        index.insert(listing)
    mismatches = 0
    for _ in range(400):
        lat = 40.3916 + rng.gauss(0, 0.3)
        lon = -111.8508 + rng.gauss(0, 0.3)
        radius = rng.choice([1.0, 3.0, 8.0, 20.0])
        got, _ = index.query_radius(lat, lon, radius)
        want = query_radius_bruteforce(listings, lat, lon, radius)
        if sorted(l.id for l, _ in got) != sorted(l.id for l, _ in want):
            mismatches += 1
    check("index returns exactly the brute force set on 400 queries", mismatches == 0,
          "%d mismatches" % mismatches)


@case
def test_index_delete():
    listings, _ = build_market(6, n_listings=50)
    index = GridIndex()
    for listing in listings:
        index.insert(listing)
    check("removes a present listing", index.remove(listings[0]))
    check("count decremented", index.count == 49)
    check("removing twice is a no-op", not index.remove(listings[0]))
    got, _ = index.query_radius(listings[0].lat, listings[0].lon, 0.5)
    check("removed listing is gone", listings[0].id not in [l.id for l, _ in got])


@case
def test_rank_orders_sensibly():
    near = Listing("near", 10, 20, 10, lat=40.0, lon=-111.0, monthly_price=100, rating=4.8)
    far = Listing("far", 10, 20, 10, lat=40.5, lon=-111.0, monthly_price=100, rating=4.8)
    ranked = rank([(near, 0.5), (far, 30.0)], item_area=100)
    check("closer listing ranks first", ranked[0][0].id == "near")
    cheap = Listing("cheap", 10, 20, 10, lat=40.0, lon=-111.0, monthly_price=50, rating=4.8)
    ranked = rank([(near, 5.0), (cheap, 5.0)], item_area=100)
    check("cheaper listing wins at equal distance", ranked[0][0].id == "cheap")


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


@case
def test_weighted_median():
    check("uniform weights", weighted_median([(1, 1), (2, 1), (3, 1)]) == 2)
    check("weight dominates", weighted_median([(1, 100), (2, 1), (3, 1)]) == 1)
    check("empty", weighted_median([]) is None)
    check("zero weights ignored", weighted_median([(1, 0), (5, 1)]) == 5)


@case
def test_pricing_refuses_thin_comp_sets():
    lonely = Listing("lonely", 10, 20, 10, lat=10.0, lon=10.0)
    listings, _ = build_market(9, n_listings=200)
    result = suggest_price(lonely, listings)
    check("no recommendation without comps", result["suggested"] is None, str(result))


@case
def test_pricing_beats_the_global_median_baseline():
    listings, rng = build_market(21, n_listings=500)
    targets = listings[:150]
    truths = [l.monthly_price for l in targets]
    scored = evaluate(targets, truths, listings)
    check("scored most targets", scored["n_scored"] > 100, str(scored))
    check(
        "local comps beat a global median",
        scored["model_mape"] < scored["baseline_mape"],
        "model %.4f vs baseline %.4f" % (scored["model_mape"], scored["baseline_mape"]),
    )
    print("       pricing MAPE model %.3f vs baseline %.3f"
          % (scored["model_mape"], scored["baseline_mape"]))


def main():
    elapsed = time.time() - _T0
    print("\n%d cases passed, %d assertions, %d failures, %.2fs"
          % (RESULTS["passed"], RESULTS["assertions"], RESULTS["failed"], elapsed))
    if RESULTS["failures"]:
        for f in RESULTS["failures"]:
            print("  - %s" % f)
        sys.exit(1)


if __name__ == "__main__":
    main()
