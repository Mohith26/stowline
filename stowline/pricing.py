"""What should a host charge for a space?

Hosts on a peer-to-peer storage marketplace are not professionals. They have no
idea whether their 20 x 20 garage in a given zip code is a $90 space or a $260
space, and a bad guess costs the marketplace real money in both directions: too
high and the listing never books, too low and the host churns out once they
figure it out.

The estimator here is a locally weighted median of comparable listings' price
per square foot. Weighted median rather than weighted mean because storage
comps have a long right tail (climate-controlled indoor units sitting in the
same comp set as an open dirt lot), and a single outlier should not drag a
recommendation. Then a size adjustment, because price per square foot is not
constant in size, and a utilization multiplier for how full the host's local
market currently is.

`evaluate` scores the whole thing against a held-out ground truth instead of
just asserting it looks sensible.
"""

import math

from .search import haversine_mi


def weighted_median(pairs):
    """pairs: iterable of (value, weight). Returns the weighted median."""
    items = sorted((v, w) for v, w in pairs if w > 0)
    if not items:
        return None
    total = sum(w for _, w in items)
    half = total / 2.0
    acc = 0.0
    for value, weight in items:
        acc += weight
        if acc >= half:
            return value
    return items[-1][0]


def comp_weights(target, comps, distance_bandwidth_mi=3.0, size_bandwidth=0.45):
    """Gaussian kernel on distance, log-ratio kernel on size.

    Size similarity is measured on log area, so a 200 sq ft space is as far
    from 100 as it is from 400. Using raw square-foot difference would make
    every small space look similar to every other small space and nothing look
    similar to a big one.
    """
    out = []
    for comp in comps:
        dist = haversine_mi(target.lat, target.lon, comp.lat, comp.lon)
        w_dist = math.exp(-0.5 * (dist / distance_bandwidth_mi) ** 2)
        log_ratio = math.log(max(comp.area, 1) / float(max(target.area, 1)))
        w_size = math.exp(-0.5 * (log_ratio / size_bandwidth) ** 2)
        out.append((comp, w_dist * w_size, dist))
    return out


def size_adjustment(area_sqft, reference_sqft=200.0, exponent=-0.05):
    """Bigger spaces rent for less per square foot.

    A power law with a negative exponent, which is the usual shape for a
    quantity discount. The default is swept in experiments.py rather than
    assumed, and the sweep produced the one genuinely counterintuitive result
    in this project: the market these listings are drawn from is built with a
    -0.15 exponent, and using -0.15 here makes the estimate measurably worse
    than -0.05. The comp kernel already weights on log size similarity, so the
    comps that survive it are mostly the same size as the target and their
    prices already carry the discount. Applying the full curve on top double
    counts it. Matching the data generating process is not the same thing as
    minimising error through a pipeline that has already conditioned on the
    same variable.
    """
    return (area_sqft / reference_sqft) ** exponent


def suggest_price(target, comps, utilization=0.5, min_effective_comps=3.0, exponent=-0.05):
    """Returns a dict with the recommendation and enough detail to explain it.

    `utilization` is how full comparable local supply currently is. The
    multiplier is deliberately gentle and clipped: an automated pricing
    recommendation that can swing a host's income by 2x on a noisy occupancy
    signal is a support ticket, not a feature.
    """
    weighted = comp_weights(target, comps)
    effective = sum(w for _, w, _ in weighted)
    if effective < min_effective_comps:
        return {
            "suggested": None,
            "reason": "only %.2f effective comps nearby, below the %.1f floor"
            % (effective, min_effective_comps),
            "effective_comps": effective,
        }
    ppsf = weighted_median(
        (comp.monthly_price / float(comp.area), w) for comp, w, _ in weighted
    )
    base = ppsf * target.area * size_adjustment(target.area, exponent=exponent)
    multiplier = 1.0 + 0.30 * (utilization - 0.5) / 0.5
    multiplier = max(0.85, min(1.30, multiplier))
    return {
        "suggested": round(base * multiplier, 2),
        "base": round(base, 2),
        "ppsf": round(ppsf, 4),
        "multiplier": round(multiplier, 3),
        "effective_comps": round(effective, 2),
        "reason": "ok",
    }


def evaluate(targets, truths, comp_pool, exponent=-0.05):
    """Mean absolute percentage error against a held-out truth price.

    Baseline is the global median price per square foot applied to the target
    area, which is what you would ship if you had no location model at all. A
    recommendation engine that cannot beat that baseline is not worth its
    complexity.
    """
    global_ppsf = weighted_median(
        (comp.monthly_price / float(comp.area), 1.0) for comp in comp_pool
    )
    model_errs = []
    base_errs = []
    skipped = 0
    for target, truth in zip(targets, truths):
        comps = [c for c in comp_pool if c.id != target.id]
        result = suggest_price(target, comps, utilization=0.5, exponent=exponent)
        if result["suggested"] is None:
            skipped += 1
            continue
        model_errs.append(abs(result["suggested"] - truth) / truth)
        base_errs.append(abs(global_ppsf * target.area - truth) / truth)
    n = len(model_errs)
    return {
        "n_scored": n,
        "n_skipped": skipped,
        "model_mape": sum(model_errs) / n if n else None,
        "baseline_mape": sum(base_errs) / n if n else None,
    }
