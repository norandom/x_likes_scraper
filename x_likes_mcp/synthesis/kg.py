"""In-memory mini knowledge graph for the synthesis-report feature.

The synthesis pipeline mines entities (handles, hashtags, domains, concepts)
and tweet IDs from round-1 hits, then assembles them into a small graph the
multihop fan-out, the mindmap renderer, and the DSPy synthesis pass all
read. The graph is intentionally minimal: a ``dict[str, Node]`` plus a
``list[Edge]`` and a few accessors. There is no ``networkx`` dependency
and nothing is persisted to disk.

Node IDs are namespaced strings so callers from different modules can
co-author the same graph without colliding (a handle and a tweet share
the namespace by virtue of their prefix). The module-level ID helpers
canonicalize their input the same way every time, which is what gives
the "two builds on the same input produce identical graphs" guarantee
its teeth.

See ``.kiro/specs/synthesis-report/design.md`` (``kg`` component) and
requirements 5.3 + 5.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "KG",
    "Edge",
    "EdgeKind",
    "Node",
    "NodeKind",
    "concept_id",
    "domain_id",
    "handle_id",
    "hashtag_id",
    "query_id",
    "serialize_kg",
    "tweet_id",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeKind(StrEnum):
    """The six node kinds the synthesis KG supports."""

    QUERY = "query"
    TWEET = "tweet"
    HANDLE = "handle"
    HASHTAG = "hashtag"
    DOMAIN = "domain"
    CONCEPT = "concept"


class EdgeKind(StrEnum):
    """The four edge kinds the synthesis KG supports."""

    AUTHORED_BY = "authored_by"
    CITES = "cites"
    MENTIONS = "mentions"
    RECALL_FOR = "recall_for"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A graph node keyed by a namespaced ID.

    Identity is the ``id`` field; ``label`` is the human-readable form
    rendered in the mindmap. ``weight`` is an additive ranking signal —
    ``KG.add_node`` sums weights when the same id is added more than
    once so callers can model "this handle was mentioned three times" as
    three separate ``add_node`` calls of weight 1.0.
    """

    id: str
    kind: NodeKind
    label: str
    weight: float


@dataclass(frozen=True)
class Edge:
    """A directed edge between two node IDs with a typed relation.

    Edges are directed: ``neighbors(src)`` returns destinations of all
    outgoing edges from ``src``. Identity for dedupe purposes is the
    full ``(src, dst, kind)`` tuple — adding the same edge twice is a
    no-op.
    """

    src: str
    dst: str
    kind: EdgeKind


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

_CONCEPT_SEPARATOR_RUN = re.compile(r"[^0-9a-z]+")


def tweet_id(raw: str) -> str:
    """Return the namespaced ID for a tweet.

    ``raw`` is the platform-side numeric tweet ID, treated as an opaque
    string so the namespace is not constrained to platforms that emit
    pure integers.
    """

    return f"tweet:{raw}"


def handle_id(handle: str) -> str:
    """Return the namespaced ID for a screen-name / handle.

    Strips a leading ``@`` if present and lowercases so ``@Tom_Doerr``
    and ``tom_doerr`` collapse onto the same node.
    """

    return f"handle:{handle.lstrip('@').lower()}"


def hashtag_id(tag: str) -> str:
    """Return the namespaced ID for a hashtag.

    Strips a leading ``#`` if present and lowercases. ``#AI`` and
    ``ai`` therefore share the same node.
    """

    return f"hashtag:{tag.lstrip('#').lower()}"


def domain_id(host: str) -> str:
    """Return the namespaced ID for a URL host.

    Lowercases so DNS-style case insensitivity is preserved at the
    graph level. ``Example.COM`` and ``example.com`` resolve to the
    same node.
    """

    return f"domain:{host.lower()}"


def concept_id(phrase: str) -> str:
    """Return the namespaced ID for a free-form concept phrase.

    Lowercases the phrase, drops every non-alphanumeric character that
    is not whitespace, and collapses runs of whitespace into a single
    underscore. ``"AI Pentesting!"`` -> ``"concept:ai_pentesting"``.
    """

    # Lowercase first so the regex character class can stay ASCII-only.
    # Then collapse any run of "non-id" characters (whitespace,
    # punctuation, hyphens) into a single underscore. Trim leading and
    # trailing underscores so ``"  foo  "`` -> ``"concept:foo"``.
    snake = _CONCEPT_SEPARATOR_RUN.sub("_", phrase.lower()).strip("_")
    return f"concept:{snake}"


def query_id() -> str:
    """Return the singleton root-node ID for the user query."""

    return "query:root"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class KG:
    """In-memory directed multigraph keyed by namespaced node ID.

    The graph is backed by a ``dict[str, Node]`` for O(1) node lookup
    and a ``list[Edge]`` for ordered, dedupe-friendly edges. Nothing is
    persisted: every instance lives only as long as the orchestrator
    keeps the reference.

    Behavioral contracts pinned by the test suite:

    - ``add_node`` accumulates weight on duplicate IDs and keeps the
      first ``label``/``kind`` (immutable identity).
    - ``add_edge`` dedupes identical ``(src, dst, kind)`` tuples.
    - ``top_entities`` sorts by ``-weight`` then by ``id`` ascending
      for deterministic ordering across runs.
    - ``neighbors`` treats the graph as directed and returns
      destination nodes for outgoing edges.
    """

    __slots__ = ("_edge_keys", "_edges", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        # Tracks dedupe keys without mutating the public edge order.
        self._edge_keys: set[tuple[str, str, EdgeKind]] = set()

    # -- mutation ---------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Insert or accumulate a node.

        First insertion stores the node verbatim. Subsequent inserts of
        the same ``id`` add the new ``weight`` to the existing weight
        and leave ``label``/``kind`` untouched, so a caller that
        observes a handle three times (each at weight 1.0) ends up with
        a single node of weight 3.0.
        """

        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            return
        # Identity (label/kind) is fixed by the first add; only weight
        # accumulates. ``Node`` is frozen, so build a fresh instance.
        self._nodes[node.id] = Node(
            id=existing.id,
            kind=existing.kind,
            label=existing.label,
            weight=existing.weight + node.weight,
        )

    def add_edge(self, edge: Edge) -> None:
        """Append an edge unless an identical one already exists.

        Identity is the ``(src, dst, kind)`` tuple. Duplicate
        ``add_edge`` calls are silently dropped so ``neighbors`` stays
        stable when callers re-emit the same edge from independent
        passes (e.g. round-1 and round-2 mining the same handle on the
        same tweet).
        """

        key = (edge.src, edge.dst, edge.kind)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._edges.append(edge)

    # -- queries ----------------------------------------------------------

    def nodes(self) -> list[Node]:
        """Return every node in insertion order.

        Read-only snapshot — mutating the returned list does not change
        the graph. The orchestrator's relevance filter and serialization
        code use this in preference to the private ``_nodes`` dict.
        """

        return list(self._nodes.values())

    def edges(self) -> list[Edge]:
        """Return every edge in insertion order.

        Read-only snapshot — same contract as :meth:`nodes`.
        """

        return list(self._edges)

    def top_entities(self, kind: NodeKind, n: int) -> list[Node]:
        """Return up to ``n`` nodes of ``kind`` sorted by weight desc.

        Ties on ``weight`` are broken by ``id`` ascending so the result
        order does not depend on insertion order. ``n`` clamps at the
        number of nodes of that kind; the returned list is always at
        most ``n`` long, possibly empty.
        """

        if n <= 0:
            return []
        candidates = [node for node in self._nodes.values() if node.kind is kind]
        candidates.sort(key=lambda node: (-node.weight, node.id))
        return candidates[:n]

    def neighbors(
        self,
        node_id: str,
        edge_kind: EdgeKind | None = None,
    ) -> list[Node]:
        """Return destination nodes reachable via outgoing edges.

        If ``edge_kind`` is ``None`` every outgoing edge from
        ``node_id`` contributes its destination; otherwise only edges
        of that kind do. Destinations are returned in edge-insertion
        order. Unknown ``node_id`` (or a known id with no outgoing
        edges) returns an empty list — callers can treat that as
        "isolated" without a membership check.
        """

        if node_id not in self._nodes:
            return []
        out: list[Node] = []
        seen: set[str] = set()
        for edge in self._edges:
            if edge.src != node_id:
                continue
            if edge_kind is not None and edge.kind is not edge_kind:
                continue
            dst_node = self._nodes.get(edge.dst)
            if dst_node is None or dst_node.id in seen:
                continue
            seen.add(dst_node.id)
            out.append(dst_node)
        return out


def serialize_kg(kg: KG) -> dict[str, list[dict[str, object]]]:
    """Return a JSON-friendly snapshot of ``kg``'s nodes and edges.

    Shape::

        {
          "nodes": [{"id": "tweet:1", "kind": "tweet", "label": "...", "weight": 1.0}, ...],
          "edges": [{"src": "query:root", "dst": "tweet:1", "kind": "recall_for"}, ...],
        }

    Determinism: nodes sort by id ascending; edges follow insertion order
    (matches what ``add_edge`` records). Two builds on the same input
    therefore produce byte-identical JSON.
    """

    nodes_payload: list[dict[str, object]] = [
        {
            "id": node.id,
            "kind": str(node.kind.value),
            "label": node.label,
            "weight": node.weight,
        }
        for node in sorted(kg._nodes.values(), key=lambda n: n.id)
    ]
    edges_payload: list[dict[str, object]] = [
        {"src": edge.src, "dst": edge.dst, "kind": str(edge.kind.value)} for edge in kg._edges
    ]
    return {"nodes": nodes_payload, "edges": edges_payload}
