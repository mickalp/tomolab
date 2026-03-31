from __future__ import annotations
from collections import defaultdict, deque

from .model import Pipeline, NodeInstance
from .types import NodeResult
from .context import RunContext
from .registry import NodeRegistry


class PipelineExecutor:
    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry

    def execute(self, pipeline: Pipeline, context: RunContext) -> dict[str, NodeResult]:
        ordered_nodes = self._topological_sort(pipeline)
        results: dict[str, NodeResult] = {}

        for node in ordered_nodes:
            if not node.enabled:
                continue

            impl = self.registry.create(node.type_name)
            inputs = self._collect_inputs(node.id, pipeline, results)

            errors = impl.validate(node.params, inputs)
            if errors:
                raise ValueError(f"Validation failed for node '{node.name}': {errors}")

            result = impl.run(
                node_id=node.id,
                params=node.params,
                inputs=inputs,
                context=context,
            )
            results[node.id] = result

        return results

    def _collect_inputs(self, node_id: str, pipeline: Pipeline, results: dict[str, NodeResult]):
        gathered = {}
        for edge in pipeline.edges:
            if edge.target_node == node_id:
                source_result = results[edge.source_node]
                gathered[edge.target_input] = source_result.outputs[edge.source_output]
        return gathered

    def _topological_sort(self, pipeline: Pipeline) -> list[NodeInstance]:
        node_map = {n.id: n for n in pipeline.nodes}
        indegree = {n.id: 0 for n in pipeline.nodes}
        outgoing = defaultdict(list)

        for e in pipeline.edges:
            outgoing[e.source_node].append(e.target_node)
            indegree[e.target_node] += 1

        queue = deque(node_map[nid] for nid, deg in indegree.items() if deg == 0)
        ordered = []

        while queue:
            node = queue.popleft()
            ordered.append(node)

            for target in outgoing[node.id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(node_map[target])

        if len(ordered) != len(pipeline.nodes):
            raise ValueError("Pipeline contains a cycle or invalid graph.")

        return ordered
