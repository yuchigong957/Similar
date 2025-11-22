"""Utility to rename and cut ONNX graphs in a stable way.

The tool keeps node-name renaming and graph cutting separated so that
pre-renamed models can still be split safely.  It builds connections
based on tensor flow rather than relying on fragile positional indices,
which makes the path search resilient to node-name changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Set

import onnx
from onnx import helper


def _ensure_node_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give every node a stable, unique name.

    The function keeps existing names untouched and only fills in missing
    ones.  Names follow the pattern ``M_<OpType>_<index>`` so they can be
    used consistently by other tools.  Returning the model itself makes it
    easy to chain this helper.
    """

    op_type_counts: Dict[str, int] = defaultdict(int)
    for node in model.graph.node:
        if node.name:
            continue
        idx = op_type_counts[node.op_type]
        node.name = f"M_{node.op_type}_{idx}"
        op_type_counts[node.op_type] += 1
    return model


def _build_adjacency(model: onnx.ModelProto) -> Dict[str, Set[str]]:
    """Create a producer -> consumer adjacency map using tensor links."""

    tensor_producer: Dict[str, str] = {}
    for node in model.graph.node:
        for out_name in node.output:
            tensor_producer[out_name] = node.name

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for node in model.graph.node:
        for input_name in node.input:
            producer = tensor_producer.get(input_name)
            if producer:
                adjacency[producer].add(node.name)
    return adjacency


def _resolve_node_names(model: onnx.ModelProto, hints: Sequence[str]) -> List[str]:
    """Resolve user provided node names or partial hints to actual names.

    This is tolerant to prior renaming: if a hint is not an exact match it
    will try to find a unique node whose name contains the hint.
    """

    available = {node.name for node in model.graph.node}
    resolved: List[str] = []

    for hint in hints:
        if hint in available:
            resolved.append(hint)
            continue

        candidates = [name for name in available if hint in name]
        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue

        message = (
            f"Node name '{hint}' not found in model; "
            "provide an exact node name or a unique substring."
        )
        raise ValueError(message)

    return resolved


def _collect_path_nodes(
    adjacency: Dict[str, Set[str]], start_nodes: Sequence[str], end_nodes: Sequence[str]
) -> Set[str]:
    """Return all nodes on every path from ``start_nodes`` to ``end_nodes``.

    A breadth-first traversal ensures we explore the full downstream area
    while avoiding infinite loops.  If no path is found, a ``ValueError``
    with helpful diagnostics is raised so the caller can adjust the node
    names.
    """

    end_set = set(end_nodes)
    visited: Set[str] = set()
    reachable: Set[str] = set()

    queue = deque(start_nodes)
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        reachable.add(current)

        for nxt in adjacency.get(current, set()):
            queue.append(nxt)

    if not end_set.intersection(reachable):
        raise ValueError(
            "[ERROR] 没找到从 {} 到 {} 的路径，请检查 unified_names 后的节点名是否正确，或者这些节点之间本来无连接。".format(
                start_nodes, end_nodes
            )
        )

    # Trim unreachable tails that are not on the path to the requested ends.
    reverse_adj: Dict[str, Set[str]] = defaultdict(set)
    for producer, consumers in adjacency.items():
        for consumer in consumers:
            reverse_adj[consumer].add(producer)

    nodes_on_path: Set[str] = set()
    queue = deque(end_nodes)
    while queue:
        current = queue.popleft()
        if current not in reachable or current in nodes_on_path:
            continue
        nodes_on_path.add(current)
        for parent in reverse_adj.get(current, set()):
            queue.append(parent)

    return nodes_on_path


def _filter_graph(model: onnx.ModelProto, keep_nodes: Set[str]) -> onnx.ModelProto:
    """Create a subgraph containing only ``keep_nodes`` and needed tensors."""

    keep_nodes_set = set(keep_nodes)
    new_model = onnx.ModelProto()
    new_model.ir_version = model.ir_version
    new_model.producer_name = model.producer_name
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version
    new_model.doc_string = model.doc_string

    new_graph = helper.make_graph([], model.graph.name, [], [])

    # Copy nodes
    for node in model.graph.node:
        if node.name in keep_nodes_set:
            new_graph.node.append(deepcopy(node))

    # Identify tensors required by the kept nodes
    produced_tensors: Set[str] = set()
    needed_inputs: Set[str] = set()
    for node in new_graph.node:
        produced_tensors.update(node.output)
        for inp in node.input:
            if inp not in produced_tensors:
                needed_inputs.add(inp)

    # Preserve graph inputs that are still required
    input_map = {i.name: i for i in model.graph.input}
    for name in needed_inputs:
        if name in input_map:
            new_graph.input.append(deepcopy(input_map[name]))

    # Keep initializers referenced by the subgraph
    init_map = {init.name: init for init in model.graph.initializer}
    for name in needed_inputs:
        if name in init_map:
            new_graph.initializer.append(deepcopy(init_map[name]))

    # Decide outputs: use original graph outputs that are produced by kept nodes;
    # if none match, fall back to outputs of the end nodes.
    output_map = {o.name: o for o in model.graph.output}
    produced_outputs = [t for t in produced_tensors if t in output_map]
    if produced_outputs:
        for name in produced_outputs:
            new_graph.output.append(deepcopy(output_map[name]))
    else:
        for node in new_graph.node:
            for out_name in node.output:
                if out_name not in output_map:
                    # fabricate a minimal ValueInfo for missing outputs
                    new_graph.output.append(helper.make_tensor_value_info(out_name, onnx.TensorProto.UNDEFINED, None))

    new_model.graph.CopyFrom(new_graph)
    return new_model


def cut_or_remove_node(
    model_path: str, start_nodes: Sequence[str], end_nodes: Sequence[str], mode: int = 0
) -> onnx.ModelProto:
    """Cut or remove nodes between ``start_nodes`` and ``end_nodes``.

    ``mode=0`` keeps only the nodes on the path (i.e., cut out a subgraph).
    ``mode=1`` removes the nodes on the path and keeps everything else.
    """

    model = onnx.load(model_path)
    _ensure_node_names(model)

    start_nodes = _resolve_node_names(model, start_nodes)
    end_nodes = _resolve_node_names(model, end_nodes)

    adjacency = _build_adjacency(model)
    nodes_on_path = _collect_path_nodes(adjacency, start_nodes, end_nodes)

    if mode == 0:
        keep_nodes = nodes_on_path
    elif mode == 1:
        keep_nodes = {node.name for node in model.graph.node if node.name not in nodes_on_path}
    else:
        raise ValueError("mode must be 0 (keep path) or 1 (remove path)")

    return _filter_graph(model, keep_nodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut or remove parts of an ONNX model by node names")
    parser.add_argument("model", help="Path to the ONNX model")
    parser.add_argument("start_nodes", nargs="+", help="Start node names or substrings")
    parser.add_argument("end_nodes", nargs="+", help="End node names or substrings")
    parser.add_argument("--mode", type=int, default=0, help="0 to keep the path, 1 to remove it")
    parser.add_argument("--output", default=None, help="Where to save the processed model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_model = cut_or_remove_node(args.model, args.start_nodes, args.end_nodes, args.mode)
    output_path = args.output or str(Path(args.model).with_name(Path(args.model).stem + "_cut.onnx"))
    onnx.save(new_model, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
