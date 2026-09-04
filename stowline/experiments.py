"""Every number in the README comes from here.

Run a single section:      python -m stowline.experiments funnel
Combine into results.json: python -m stowline.experiments combine

Split into sections because the exhaustive oracle is exponential and running
everything in one process is a good way to wait ten minutes for a number you
could have had in twenty seconds.
"""

import json
import os
import random
import sys
import time

from .booking import Listing, ListingState, quote
from .generate import build_market, random_instance, ITEM_TYPES, SPACE_TYPES
from .placement import Reservation, solve, solve_exhaustive
from .pricing import evaluate, suggest_price
from .search import GridIndex, query_radius_bruteforce
from .timeline import peak_area

OUT_DIR = "results"


def _save(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "part_%s.json" % name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote %s" % path)
    print(json.dumps(payload, indent=2, sort_keys=True))


def funnel(seed=1, n_listings=60, requests_per_listing=40):
    """Where do booking requests actually die?

    Fills a population of listings with random demand and records, for every
    rejected request, which stage caught it. The number worth looking at is
    `area_ok_but_no_placement`: requests that had the square footage available
    and still had nowhere legal to go.
    """
    rng = random.Random(seed)
    stages = {"dimensions": 0, "area": 0, "placement": 0}
    accepted = 0
    total = 0
    repack_would_have_fit = 0
    placement_rejections = 0
    t0 = time.time()
    for li in range(n_listings):
        label, w, d, door = SPACE_TYPES[li % len(SPACE_TYPES)]
        listing = Listing("L%03d" % li, width=w, depth=d, door_width=door)
        state = ListingState(listing)
        for ri in range(requests_per_listing):
            name, iw, idd, rot = ITEM_TYPES[rng.randrange(len(ITEM_TYPES))]
            start = rng.randrange(0, 300)
            end = start + rng.choice([30, 60, 90, 180, 365 - start if start < 100 else 120])
            end = min(end, 365)
            if end <= start:
                continue
            req = Reservation("L%03dR%03d" % (li, ri), iw, idd, start, end, rot)
            total += 1
            decision = state.try_book(req, check_repack=True, budget=120000)
            if decision.accepted:
                accepted += 1
            else:
                stages[decision.stage] += 1
                if decision.stage == "placement":
                    placement_rejections += 1
                    if decision.repack_would_fit:
                        repack_would_have_fit += 1
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "listings": n_listings,
        "requests": total,
        "accepted": accepted,
        "accept_rate": round(accepted / float(total), 4),
        "rejected_at_dimensions": stages["dimensions"],
        "rejected_at_area": stages["area"],
        "rejected_at_placement": stages["placement"],
        "area_ok_but_no_placement": stages["placement"],
        "share_of_rejections_that_are_geometric": round(
            stages["placement"] / float(max(1, total - accepted)), 4
        ),
        "placement_rejections_a_repack_would_have_saved": repack_would_have_fit,
        "repack_recovery_rate": round(
            repack_would_have_fit / float(max(1, placement_rejections)), 4
        ),
        "seconds": round(elapsed, 2),
        "decisions_per_second": round(total / elapsed, 1),
    }


def oracle(seed=2, trials=400):
    """How often does the fast corner-point search miss a solution that a
    complete search over every integer position can find?

    For plain rectangle packing the answer should be never: any packing can be
    slid down and left until every item touches an edge or a neighbour, and
    corner-point placement considers exactly those positions. The exit-channel
    constraint breaks that argument, because sliding an item toward the door is
    the move that puts it in somebody else's way. This measures the cost.
    """
    rng = random.Random(seed)
    resolved = agree = fast_missed = undecided = 0
    impossible = 0
    fast_nodes = 0
    slow_nodes = 0
    t0 = time.time()
    for _ in range(trials):
        w, d = rng.choice([(10, 12), (12, 14), (10, 16), (12, 12)])
        items = random_instance(rng, w, d, rng.randint(2, 4), horizon=60, max_side=7)
        fast, _, fn = solve(w, d, items, budget=80000)
        slow, _, sn = solve_exhaustive(w, d, items, budget=400000)
        fast_nodes += fn
        slow_nodes += sn
        if fast is None or slow is None:
            undecided += 1
            continue
        resolved += 1
        if fast == slow:
            agree += 1
        elif slow and not fast:
            fast_missed += 1
        else:
            impossible += 1
    return {
        "seed": seed,
        "trials": trials,
        "resolved": resolved,
        "undecided_within_budget": undecided,
        "agree": agree,
        "fast_path_missed_a_real_solution": fast_missed,
        "fast_path_found_what_oracle_could_not": impossible,
        "miss_rate": round(fast_missed / float(max(1, resolved)), 4),
        "mean_nodes_fast": round(fast_nodes / float(trials), 1),
        "mean_nodes_exhaustive": round(slow_nodes / float(trials), 1),
        "node_ratio": round(slow_nodes / float(max(1, fast_nodes)), 1),
        "seconds": round(time.time() - t0, 2),
    }


def search_bench(seed=3, n_listings=20000, queries=800):
    listings, rng = build_market(seed, n_listings=n_listings)
    results = {}
    for cell in (0.01, 0.02, 0.05, 0.1, 0.25):
        index = GridIndex(cell_deg=cell)
        for listing in listings:
            index.insert(listing)
        probes = []
        for _ in range(queries):
            probes.append(
                (
                    40.3916 + rng.gauss(0, 0.3),
                    -111.8508 + rng.gauss(0, 0.3),
                    rng.choice([1.0, 3.0, 10.0]),
                )
            )
        t0 = time.time()
        examined = 0
        hits = 0
        for lat, lon, radius in probes:
            got, ex = index.query_radius(lat, lon, radius)
            examined += ex
            hits += len(got)
        idx_time = time.time() - t0
        results["cell_%g" % cell] = {
            "seconds": round(idx_time, 3),
            "queries_per_second": round(queries / idx_time, 1),
            "mean_candidates_examined": round(examined / float(queries), 1),
            "mean_hits": round(hits / float(queries), 1),
        }
    t0 = time.time()
    bf_probes = probes[:200]
    for lat, lon, radius in bf_probes:
        query_radius_bruteforce(listings, lat, lon, radius)
    bf_qps = len(bf_probes) / (time.time() - t0)
    best = max(results.items(), key=lambda kv: kv[1]["queries_per_second"])
    return {
        "seed": seed,
        "listings": n_listings,
        "queries": queries,
        "by_cell_size": results,
        "brute_force_queries_per_second": round(bf_qps, 1),
        "best_cell_deg": best[0],
        "speedup_over_brute_force": round(best[1]["queries_per_second"] / bf_qps, 1),
    }


def search_exactness(seed=4, n_listings=6000, queries=1500):
    listings, rng = build_market(seed, n_listings=n_listings)
    index = GridIndex(cell_deg=0.05)
    for listing in listings:
        index.insert(listing)
    mismatched = 0
    for _ in range(queries):
        lat = 40.3916 + rng.gauss(0, 0.4)
        lon = -111.8508 + rng.gauss(0, 0.4)
        radius = rng.choice([0.5, 2.0, 7.0, 25.0])
        got, _ = index.query_radius(lat, lon, radius)
        want = query_radius_bruteforce(listings, lat, lon, radius)
        if sorted(l.id for l, _ in got) != sorted(l.id for l, _ in want):
            mismatched += 1
    return {
        "seed": seed,
        "listings": n_listings,
        "queries": queries,
        "queries_disagreeing_with_brute_force": mismatched,
    }


def pricing(seed=5, n_listings=1200, n_targets=400):
    listings, _ = build_market(seed, n_listings=n_listings)
    targets = listings[:n_targets]
    truths = [l.monthly_price for l in targets]
    by_exponent = {}
    for exponent in (0.0, -0.05, -0.10, -0.15, -0.20, -0.30):
        scored = evaluate(targets, truths, listings, exponent=exponent)
        by_exponent["%.2f" % exponent] = round(scored["model_mape"], 4)
    scored = evaluate(targets, truths, listings, exponent=-0.05)
    best = min(by_exponent.items(), key=lambda kv: kv[1])
    return {
        "seed": seed,
        "listings": n_listings,
        "targets_scored": scored["n_scored"],
        "targets_skipped_thin_comps": scored["n_skipped"],
        "model_mape": round(scored["model_mape"], 4),
        "global_median_baseline_mape": round(scored["baseline_mape"], 4),
        "improvement_vs_baseline": round(
            1 - scored["model_mape"] / scored["baseline_mape"], 4
        ),
        "mape_by_size_exponent": by_exponent,
        "best_exponent": best[0],
    }


def throughput(seed=6, n=4000):
    """How fast is a single booking decision on a listing that already has
    inventory on it? This is the number that decides whether the placement
    search can sit in the request path or has to move behind a queue."""
    rng = random.Random(seed)
    listing = Listing("L", width=20, depth=30, door_width=20)
    state = ListingState(listing)
    for i in range(6):
        item = random_instance(rng, 20, 30, 1, horizon=365, max_side=10)[0]
        state.try_book(
            Reservation("seed%d" % i, item.w, item.d, item.start, item.end, item.rotatable),
            check_repack=False,
        )
    requests = []
    for i in range(n):
        item = random_instance(rng, 20, 30, 1, horizon=365, max_side=10)[0]
        requests.append(
            Reservation("Q%d" % i, item.w, item.d, item.start, item.end, item.rotatable)
        )
    t0 = time.time()
    accepted = 0
    for req in requests:
        decision = quote(
            listing, state.reservations, state.placement, req, check_repack=False
        )
        if decision.accepted:
            accepted += 1
    elapsed = time.time() - t0
    return {
        "seed": seed,
        "existing_reservations_on_listing": len(state.reservations),
        "decisions": n,
        "accepted": accepted,
        "seconds": round(elapsed, 3),
        "decisions_per_second": round(n / elapsed, 1),
        "mean_microseconds_per_decision": round(elapsed / n * 1e6, 1),
    }


SECTIONS = {
    "funnel": funnel,
    "oracle": oracle,
    "search_bench": search_bench,
    "search_exactness": search_exactness,
    "pricing": pricing,
    "throughput": throughput,
}


def combine():
    out = {}
    for name in list(SECTIONS) + ["controls"]:
        path = os.path.join(OUT_DIR, "part_%s.json" % name)
        if os.path.exists(path):
            with open(path) as fh:
                out[name] = json.load(fh)
        else:
            out[name] = None
    out["generated_by"] = "python -m stowline.experiments <section>"
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s/results.json" % OUT_DIR)
    print(json.dumps(out, indent=2, sort_keys=True))


def main():
    if len(sys.argv) < 2:
        print("sections: %s, combine" % ", ".join(sorted(SECTIONS)))
        return
    name = sys.argv[1]
    if name == "combine":
        combine()
        return
    if name not in SECTIONS:
        print("unknown section %r" % name)
        return
    _save(name, SECTIONS[name]())


if __name__ == "__main__":
    main()
