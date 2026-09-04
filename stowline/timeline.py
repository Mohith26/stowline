"""Time-axis reasoning over a set of reservations on one listing.

Reservations are half-open day intervals [start, end). Day `end` is the day the
item is gone, so an item with end == 40 occupies day 39 and not day 40.

Two things live here:

  * `active_at` / `event_days`, because you never need to check every day of a
    year. The occupancy set only changes on a start or an end, so the whole
    timeline collapses to at most 2n interesting instants.

  * `peak_area`, a cheap necessary condition. If the total square footage of
    the items present at some instant exceeds the listing's square footage,
    no arrangement can possibly work and there is no point running the
    placement search. It is only a necessary condition, never sufficient, and
    measuring exactly how far from sufficient it is turned out to be the most
    interesting number in the whole project (see README).
"""


def event_days(reservations):
    """Every day on which the active set can change."""
    days = set()
    for r in reservations:
        days.add(r.start)
        days.add(r.end)
    return sorted(days)


def active_at(reservations, day):
    return [r for r in reservations if r.start <= day < r.end]


def overlapping(reservations, target):
    """Reservations whose stay overlaps `target`'s stay at all."""
    return [
        r
        for r in reservations
        if r.id != target.id and r.start < target.end and target.start < r.end
    ]


def peak_area(reservations):
    """Maximum simultaneous square footage, by sweep over start/end events.

    Sorted so that departures at time t are applied before arrivals at time t,
    which is correct for half-open intervals: an item ending on day 40 has
    already left when an item starting on day 40 arrives.
    """
    events = []
    for r in reservations:
        events.append((r.start, 1, r.w * r.d))
        events.append((r.end, 0, -(r.w * r.d)))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = 0
    peak = 0
    for _, _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def peak_area_bruteforce(reservations):
    """Day-by-day version of `peak_area`, used as a test oracle."""
    if not reservations:
        return 0
    lo = min(r.start for r in reservations)
    hi = max(r.end for r in reservations)
    peak = 0
    for day in range(lo, hi):
        total = sum(r.w * r.d for r in reservations if r.start <= day < r.end)
        peak = max(peak, total)
    return peak


def nested_within(inner, outer):
    """True if `inner`'s stay is strictly inside `outer`'s stay.

    This is the exact condition under which one item is allowed to block
    another's exit channel: it has to arrive after the blocked item arrived and
    leave before the blocked item's last day, so the blocked item is never
    boxed in on a day it needs to move.
    """
    return inner.start > outer.start and inner.end < outer.end
