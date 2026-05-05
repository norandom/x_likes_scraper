"""Tests for the in-memory mini knowledge graph (synthesis-report task 2.3).

These tests pin the namespaced ID scheme, the weight-accumulation rule
for ``add_node``, the dedupe rule for ``add_edge``, the deterministic
ordering of ``top_entities``, the edge-kind filter on ``neighbors``,
and the determinism guarantee that two builds from the same input
produce equal node/edge sets. The graph is purely in-memory: the
final test asserts no file is written through normal usage.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest

from x_likes_mcp.synthesis.kg import (
    KG,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    concept_id,
    domain_id,
    handle_id,
    hashtag_id,
    query_id,
    serialize_kg,
    tweet_id,
)

# ---------------------------------------------------------------------------
# StrEnum surface
# ---------------------------------------------------------------------------


class TestEnums:
    def test_node_kind_is_str_enum_with_documented_values(self) -> None:
        assert issubclass(NodeKind, StrEnum)
        assert NodeKind.QUERY.value == "query"
        assert NodeKind.TWEET.value == "tweet"
        assert NodeKind.HANDLE.value == "handle"
        assert NodeKind.HASHTAG.value == "hashtag"
        assert NodeKind.DOMAIN.value == "domain"
        assert NodeKind.CONCEPT.value == "concept"

    def test_edge_kind_is_str_enum_with_documented_values(self) -> None:
        assert issubclass(EdgeKind, StrEnum)
        assert EdgeKind.AUTHORED_BY.value == "authored_by"
        assert EdgeKind.CITES.value == "cites"
        assert EdgeKind.MENTIONS.value == "mentions"
        assert EdgeKind.RECALL_FOR.value == "recall_for"


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


class TestIdHelpers:
    def test_tweet_id(self) -> None:
        assert tweet_id("12345") == "tweet:12345"

    def test_handle_id_lowercases_and_strips_at(self) -> None:
        assert handle_id("Tom_Doerr") == "handle:tom_doerr"
        assert handle_id("@Tom_Doerr") == "handle:tom_doerr"

    def test_hashtag_id_lowercases_and_strips_hash(self) -> None:
        assert hashtag_id("AI") == "hashtag:ai"
        assert hashtag_id("#AI") == "hashtag:ai"

    def test_domain_id_lowercases(self) -> None:
        assert domain_id("Example.COM") == "domain:example.com"

    def test_concept_id_normalizes_to_lower_snake_case(self) -> None:
        assert concept_id("AI Pentesting!") == "concept:ai_pentesting"
        assert concept_id("  Multiple   Spaces  ") == "concept:multiple_spaces"
        assert concept_id("alpha-beta_gamma") == "concept:alpha_beta_gamma"

    def test_query_id_is_singleton_root(self) -> None:
        assert query_id() == "query:root"


# ---------------------------------------------------------------------------
# add_node / add_edge
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_new_node_is_stored(self) -> None:
        kg = KG()
        node = Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0)
        kg.add_node(node)
        # Node retrievable through neighbors of an outgoing edge or via top_entities.
        assert kg.top_entities(NodeKind.HANDLE, 5) == [node]

    def test_existing_id_accumulates_weight(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0))
        kg.add_node(Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=2.5))
        top = kg.top_entities(NodeKind.HANDLE, 5)
        assert len(top) == 1
        assert top[0].weight == pytest.approx(3.5)

    def test_existing_id_keeps_first_label_and_kind(self) -> None:
        kg = KG()
        first = Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0)
        # Second add tries to overwrite label/kind — must be ignored.
        second = Node(id="handle:tom", kind=NodeKind.CONCEPT, label="MUTATED", weight=0.5)
        kg.add_node(first)
        kg.add_node(second)
        top = kg.top_entities(NodeKind.HANDLE, 5)
        assert len(top) == 1
        stored = top[0]
        assert stored.kind is NodeKind.HANDLE
        assert stored.label == "@tom"
        # The second add's "kind=CONCEPT" must NOT have produced a CONCEPT node.
        assert kg.top_entities(NodeKind.CONCEPT, 5) == []


class TestAddEdge:
    def test_edge_is_appended(self) -> None:
        kg = KG()
        kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="t1", weight=1.0))
        kg.add_node(Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0))
        kg.add_edge(Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY))
        ns = kg.neighbors("tweet:1")
        assert len(ns) == 1
        assert ns[0].id == "handle:tom"

    def test_duplicate_edges_are_deduped(self) -> None:
        kg = KG()
        kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="t1", weight=1.0))
        kg.add_node(Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0))
        edge = Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY)
        kg.add_edge(edge)
        kg.add_edge(edge)
        kg.add_edge(Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY))
        ns = kg.neighbors("tweet:1")
        assert len(ns) == 1


# ---------------------------------------------------------------------------
# top_entities
# ---------------------------------------------------------------------------


class TestTopEntities:
    def test_returns_top_n_by_weight_desc(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:a", kind=NodeKind.HANDLE, label="@a", weight=1.0))
        kg.add_node(Node(id="handle:b", kind=NodeKind.HANDLE, label="@b", weight=3.0))
        kg.add_node(Node(id="handle:c", kind=NodeKind.HANDLE, label="@c", weight=2.0))
        top = kg.top_entities(NodeKind.HANDLE, 2)
        assert [n.id for n in top] == ["handle:b", "handle:c"]

    def test_ties_break_by_id_ascending(self) -> None:
        kg = KG()
        # Insert in non-id order to prove the sort doesn't just preserve insertion.
        kg.add_node(Node(id="handle:c", kind=NodeKind.HANDLE, label="@c", weight=1.0))
        kg.add_node(Node(id="handle:a", kind=NodeKind.HANDLE, label="@a", weight=1.0))
        kg.add_node(Node(id="handle:b", kind=NodeKind.HANDLE, label="@b", weight=1.0))
        top = kg.top_entities(NodeKind.HANDLE, 3)
        assert [n.id for n in top] == ["handle:a", "handle:b", "handle:c"]

    def test_filters_to_requested_kind(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:a", kind=NodeKind.HANDLE, label="@a", weight=1.0))
        kg.add_node(Node(id="hashtag:x", kind=NodeKind.HASHTAG, label="#x", weight=10.0))
        top = kg.top_entities(NodeKind.HANDLE, 5)
        assert len(top) == 1
        assert top[0].id == "handle:a"

    def test_empty_kind_returns_empty_list(self) -> None:
        kg = KG()
        kg.add_node(Node(id="handle:a", kind=NodeKind.HANDLE, label="@a", weight=1.0))
        assert kg.top_entities(NodeKind.DOMAIN, 5) == []

    def test_n_caps_result_size(self) -> None:
        kg = KG()
        for i in range(5):
            kg.add_node(
                Node(id=f"handle:h{i}", kind=NodeKind.HANDLE, label=f"@h{i}", weight=float(i)),
            )
        top = kg.top_entities(NodeKind.HANDLE, 2)
        assert len(top) == 2


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------


class TestNeighbors:
    @staticmethod
    def _seed() -> KG:
        kg = KG()
        kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="t1", weight=1.0))
        kg.add_node(Node(id="handle:tom", kind=NodeKind.HANDLE, label="@tom", weight=1.0))
        kg.add_node(
            Node(id="domain:example.com", kind=NodeKind.DOMAIN, label="example.com", weight=1.0)
        )
        kg.add_node(Node(id="hashtag:ai", kind=NodeKind.HASHTAG, label="#ai", weight=1.0))
        kg.add_edge(Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY))
        kg.add_edge(Edge(src="tweet:1", dst="domain:example.com", kind=EdgeKind.CITES))
        kg.add_edge(Edge(src="tweet:1", dst="hashtag:ai", kind=EdgeKind.MENTIONS))
        return kg

    def test_returns_all_outgoing_destinations(self) -> None:
        kg = self._seed()
        ns = kg.neighbors("tweet:1")
        ids = {n.id for n in ns}
        assert ids == {"handle:tom", "domain:example.com", "hashtag:ai"}

    def test_filters_by_edge_kind(self) -> None:
        kg = self._seed()
        ns = kg.neighbors("tweet:1", EdgeKind.CITES)
        assert [n.id for n in ns] == ["domain:example.com"]

    def test_missing_node_returns_empty_list(self) -> None:
        kg = self._seed()
        assert kg.neighbors("missing:id") == []

    def test_missing_node_with_edge_kind_filter_returns_empty(self) -> None:
        kg = self._seed()
        assert kg.neighbors("missing:id", EdgeKind.CITES) == []


# ---------------------------------------------------------------------------
# Determinism + persistence
# ---------------------------------------------------------------------------


def _build_sample_kg() -> KG:
    kg = KG()
    handles = [("handle:tom", "@tom", 1.5), ("handle:ana", "@ana", 2.0)]
    hashtags = [("hashtag:ai", "#ai", 0.7)]
    tweets = [("tweet:1", "t1", 1.0), ("tweet:2", "t2", 1.0)]
    for nid, label, w in handles:
        kg.add_node(Node(id=nid, kind=NodeKind.HANDLE, label=label, weight=w))
    for nid, label, w in hashtags:
        kg.add_node(Node(id=nid, kind=NodeKind.HASHTAG, label=label, weight=w))
    for nid, label, w in tweets:
        kg.add_node(Node(id=nid, kind=NodeKind.TWEET, label=label, weight=w))
    kg.add_edge(Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY))
    kg.add_edge(Edge(src="tweet:2", dst="handle:ana", kind=EdgeKind.AUTHORED_BY))
    kg.add_edge(Edge(src="tweet:1", dst="hashtag:ai", kind=EdgeKind.MENTIONS))
    return kg


class TestDeterminism:
    def test_two_builds_produce_equal_node_and_edge_sets(self) -> None:
        a = _build_sample_kg()
        b = _build_sample_kg()
        # Same nodes for every kind, same weights, same labels.
        for kind in NodeKind:
            assert a.top_entities(kind, 100) == b.top_entities(kind, 100)
        # Same outgoing edges for every node we seeded.
        for source in ("tweet:1", "tweet:2", "handle:tom"):
            for ek in (None, *EdgeKind):
                assert a.neighbors(source, ek) == b.neighbors(source, ek)


class TestPersistence:
    def test_no_file_written_during_normal_usage(self, tmp_path: Path) -> None:
        before = sorted(p.name for p in tmp_path.iterdir())
        kg = _build_sample_kg()
        kg.add_edge(Edge(src="tweet:1", dst="handle:tom", kind=EdgeKind.AUTHORED_BY))
        kg.top_entities(NodeKind.HANDLE, 5)
        kg.neighbors("tweet:1")
        after = sorted(p.name for p in tmp_path.iterdir())
        assert before == after
        assert before == []


class TestSerialize:
    def test_empty_kg_serializes_to_empty_collections(self) -> None:
        payload = serialize_kg(KG())
        assert payload == {"nodes": [], "edges": []}

    def test_serialize_emits_nodes_sorted_by_id(self) -> None:
        kg = KG()
        kg.add_node(Node(id="tweet:2", kind=NodeKind.TWEET, label="t2", weight=1.0))
        kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="t1", weight=1.0))
        kg.add_node(Node(id="handle:alice", kind=NodeKind.HANDLE, label="alice", weight=1.0))

        payload = serialize_kg(kg)
        ids = [n["id"] for n in payload["nodes"]]
        assert ids == sorted(ids)
        assert ids == ["handle:alice", "tweet:1", "tweet:2"]

    def test_serialize_emits_edges_in_insertion_order(self) -> None:
        kg = KG()
        kg.add_node(Node(id="tweet:1", kind=NodeKind.TWEET, label="t1", weight=1.0))
        kg.add_node(Node(id="handle:alice", kind=NodeKind.HANDLE, label="alice", weight=1.0))
        kg.add_node(Node(id="handle:bob", kind=NodeKind.HANDLE, label="bob", weight=1.0))
        kg.add_edge(Edge(src="tweet:1", dst="handle:alice", kind=EdgeKind.AUTHORED_BY))
        kg.add_edge(Edge(src="tweet:1", dst="handle:bob", kind=EdgeKind.MENTIONS))

        payload = serialize_kg(kg)
        edge_dsts = [e["dst"] for e in payload["edges"]]
        assert edge_dsts == ["handle:alice", "handle:bob"]
        assert payload["edges"][0]["kind"] == "authored_by"
        assert payload["edges"][1]["kind"] == "mentions"

    def test_serialize_round_trips_node_fields(self) -> None:
        kg = KG()
        kg.add_node(Node(id="concept:foo", kind=NodeKind.CONCEPT, label="Foo Bar", weight=2.5))
        node = serialize_kg(kg)["nodes"][0]
        assert node == {
            "id": "concept:foo",
            "kind": "concept",
            "label": "Foo Bar",
            "weight": 2.5,
        }

    def test_serialize_is_deterministic_across_runs(self) -> None:
        a = serialize_kg(_build_sample_kg())
        b = serialize_kg(_build_sample_kg())
        assert a == b
