# src/kg/build_graph.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.common.schema import NbaUpliftSchema


@dataclass(frozen=True)
class KGNodeSchema:
    node_id: str = "node_id"
    node_type: str = "node_type"
    label: str = "label"
    persona_id: str = "persona_id"
    tenure_bucket: str = "tenure_bucket"
    entity_type: str = "entity_type"
    entity_id: str = "entity_id"


@dataclass(frozen=True)
class KGEdgeSchema:
    src_id: str = "src_id"
    dst_id: str = "dst_id"
    edge_type: str = "edge_type"
    weight: str = "weight"
    support: str = "support"
    persona_id: str = "persona_id"
    tenure_bucket: str = "tenure_bucket"
    entity_type: str = "entity_type"
    entity_id: str = "entity_id"


def _persona_node_id(persona_id: int | str) -> str:
    return f"persona:{int(persona_id)}"


def _tenure_node_id(tenure_bucket: str) -> str:
    return f"tenure:{tenure_bucket}"


def _entity_node_id(entity_type: str, entity_id: int | str) -> str:
    return f"entity:{entity_type}:{int(entity_id)}"


def _segment_node_id(persona_id: int | str, tenure_bucket: str) -> str:
    return f"segment:p{int(persona_id)}:t{tenure_bucket}"


def build_kg_nodes(uplift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build KG nodes (persona, tenure, entity, segment) from uplift summary.
    """
    schema = NbaUpliftSchema()
    kg_schema = KGNodeSchema()

    # Persona nodes
    persona_nodes = (
        uplift_df[[schema.persona_id]]
        .drop_duplicates()
        .assign(
            **{
                kg_schema.node_id: lambda df: df[schema.persona_id].apply(
                    _persona_node_id
                ),
                kg_schema.node_type: "persona",
                kg_schema.label: lambda df: "Persona "
                + df[schema.persona_id].astype(str),
                kg_schema.tenure_bucket: np.nan,
                kg_schema.entity_type: np.nan,
                kg_schema.entity_id: np.nan,
            }
        )[
            [
                kg_schema.node_id,
                kg_schema.node_type,
                kg_schema.label,
                kg_schema.persona_id,
                kg_schema.tenure_bucket,
                kg_schema.entity_type,
                kg_schema.entity_id,
            ]
        ]
    )

    # Tenure nodes
    tenure_nodes = (
        uplift_df[[schema.tenure_bucket]]
        .drop_duplicates()
        .assign(
            **{
                kg_schema.node_id: lambda df: df[schema.tenure_bucket].apply(
                    _tenure_node_id
                ),
                kg_schema.node_type: "tenure",
                kg_schema.label: lambda df: df[schema.tenure_bucket],
                kg_schema.persona_id: np.nan,
                kg_schema.entity_type: np.nan,
                kg_schema.entity_id: np.nan,
            }
        )[
            [
                kg_schema.node_id,
                kg_schema.node_type,
                kg_schema.label,
                kg_schema.persona_id,
                kg_schema.tenure_bucket,
                kg_schema.entity_type,
                kg_schema.entity_id,
            ]
        ]
    )

    # Entity nodes
    entity_nodes = (
        uplift_df[
            [
                schema.entity_type,
                schema.entity_id,
                schema.entity_name,
            ]
        ]
        .drop_duplicates()
        .assign(
            **{
                kg_schema.node_id: lambda df: df.apply(
                    lambda row: _entity_node_id(
                        row[schema.entity_type], row[schema.entity_id]
                    ),
                    axis=1,
                ),
                kg_schema.node_type: "entity",
                kg_schema.label: lambda df: df[schema.entity_name],
                kg_schema.persona_id: np.nan,
                kg_schema.tenure_bucket: np.nan,
            }
        )[
            [
                kg_schema.node_id,
                kg_schema.node_type,
                kg_schema.label,
                kg_schema.persona_id,
                kg_schema.tenure_bucket,
                schema.entity_type,
                schema.entity_id,
            ]
        ]
        .rename(
            columns={
                schema.entity_type: kg_schema.entity_type,
                schema.entity_id: kg_schema.entity_id,
            }
        )
    )

    # Segment nodes: persona × tenure
    segment_nodes = (
        uplift_df[[schema.persona_id, schema.tenure_bucket]]
        .drop_duplicates()
        .assign(
            **{
                kg_schema.node_id: lambda df: df.apply(
                    lambda row: _segment_node_id(
                        row[schema.persona_id], row[schema.tenure_bucket]
                    ),
                    axis=1,
                ),
                kg_schema.node_type: "segment",
                kg_schema.label: lambda df: df.apply(
                    lambda row: f"Persona {row[schema.persona_id]} | {row[schema.tenure_bucket]}",
                    axis=1,
                ),
                kg_schema.entity_type: np.nan,
                kg_schema.entity_id: np.nan,
            }
        )[
            [
                kg_schema.node_id,
                kg_schema.node_type,
                kg_schema.label,
                schema.persona_id,
                schema.tenure_bucket,
                kg_schema.entity_type,
                kg_schema.entity_id,
            ]
        ]
        .rename(
            columns={
                schema.persona_id: kg_schema.persona_id,
                schema.tenure_bucket: kg_schema.tenure_bucket,
            }
        )
    )

    nodes = pd.concat(
        [persona_nodes, tenure_nodes, entity_nodes, segment_nodes],
        axis=0,
        ignore_index=True,
    ).drop_duplicates(subset=[kg_schema.node_id])

    return nodes


def build_kg_edges(
    uplift_df: pd.DataFrame,
    min_uplift_for_edge: float = 0.0,
) -> pd.DataFrame:
    """
    Build KG edges from uplift summary:
    - persona_entity
    - tenure_entity
    - segment_entity
    """
    schema = NbaUpliftSchema()
    kg_schema = KGEdgeSchema()

    df = uplift_df.copy()

    # Filter by uplift threshold if configured
    if min_uplift_for_edge is not None and min_uplift_for_edge > 0:
        df = df[df[schema.incremental_renewal_rate] >= min_uplift_for_edge]

    # Support = total matched population
    support = df[schema.n_test_matched] + df[schema.n_control_matched]

    # persona_entity edges
    persona_edges = pd.DataFrame(
        {
            kg_schema.src_id: df[schema.persona_id].apply(_persona_node_id),
            kg_schema.dst_id: df.apply(
                lambda row: _entity_node_id(
                    row[schema.entity_type], row[schema.entity_id]
                ),
                axis=1,
            ),
            kg_schema.edge_type: "persona_entity",
            kg_schema.weight: df[schema.incremental_renewal_rate],
            kg_schema.support: support,
            kg_schema.persona_id: df[schema.persona_id],
            kg_schema.tenure_bucket: df[schema.tenure_bucket],
            kg_schema.entity_type: df[schema.entity_type],
            kg_schema.entity_id: df[schema.entity_id],
        }
    )

    # tenure_entity edges
    tenure_edges = pd.DataFrame(
        {
            kg_schema.src_id: df[schema.tenure_bucket].apply(_tenure_node_id),
            kg_schema.dst_id: df.apply(
                lambda row: _entity_node_id(
                    row[schema.entity_type], row[schema.entity_id]
                ),
                axis=1,
            ),
            kg_schema.edge_type: "tenure_entity",
            kg_schema.weight: df[schema.incremental_renewal_rate],
            kg_schema.support: support,
            kg_schema.persona_id: df[schema.persona_id],
            kg_schema.tenure_bucket: df[schema.tenure_bucket],
            kg_schema.entity_type: df[schema.entity_type],
            kg_schema.entity_id: df[schema.entity_id],
        }
    )

    # segment_entity edges
    segment_edges = pd.DataFrame(
        {
            kg_schema.src_id: df.apply(
                lambda row: _segment_node_id(
                    row[schema.persona_id], row[schema.tenure_bucket]
                ),
                axis=1,
            ),
            kg_schema.dst_id: df.apply(
                lambda row: _entity_node_id(
                    row[schema.entity_type], row[schema.entity_id]
                ),
                axis=1,
            ),
            kg_schema.edge_type: "segment_entity",
            kg_schema.weight: df[schema.incremental_renewal_rate],
            kg_schema.support: support,
            kg_schema.persona_id: df[schema.persona_id],
            kg_schema.tenure_bucket: df[schema.tenure_bucket],
            kg_schema.entity_type: df[schema.entity_type],
            kg_schema.entity_id: df[schema.entity_id],
        }
    )

    edges = pd.concat(
        [persona_edges, tenure_edges, segment_edges],
        axis=0,
        ignore_index=True,
    )

    # Optional: drop exact duplicates if any
    edges = edges.drop_duplicates(
        subset=[kg_schema.src_id, kg_schema.dst_id, kg_schema.edge_type]
    )

    return edges
