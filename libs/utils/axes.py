#########################################################################
#  Module: axes                                                         #
#  Description: The single list of experiment-matrix axes. Three separate #
#               tools (shared_node_min, expr_audit, tier2_collect) each   #
#               built their own grouping key and each of them dropped a   #
#               different axis, silently collapsing distinct measurements  #
#               onto one row (docs/90 N30, N33). Any code that groups,    #
#               dedupes or pairs measurement rows imports AXES from here   #
#               instead of writing a tuple by hand.                       #
#  Pipeline: results.csv -> (any grouping tool) -> tables                 #
#########################################################################

from __future__ import annotations

# Order matters only for display. A column absent from a given CSV is skipped
# by key_of(), so the same list serves the model sweeps and the kernel matrix.
AXES: tuple[str, ...] = (
    "host", "backend", "threads", "precision",
    "case", "scheme", "theta", "solver", "cfl",
    "nx", "ny", "nz", "n_steps",
    "topo", "r_std", "rx0", "h_ratio",
)


def present(row: dict) -> tuple[str, ...]:
    """The axes this row actually carries."""
    return tuple(a for a in AXES if a in row)


def canon(value) -> str:
    """Canonical text for an axis value.

    A numeric axis rendered differently by two writers ("0" vs "0.0", "1e3" vs
    "1000") names the SAME point; keying on the raw text puts them in different
    buckets and the pair silently never forms.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    return f"{int(f)}" if f == int(f) else repr(f)


def key_of(row: dict, axes: tuple[str, ...] | None = None,
           *, drop: tuple[str, ...] = ()) -> tuple:
    """Grouping key over every axis except those explicitly dropped.

    `drop` is how a caller says "this axis is the thing I am comparing" - and
    because it has to be named, an axis can no longer go missing by accident.
    """
    axes = axes if axes is not None else present(row)
    return tuple(canon(row.get(a, "")) for a in axes if a not in drop)
