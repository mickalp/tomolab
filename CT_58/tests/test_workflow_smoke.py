from pathlib import Path

from tomolab.workflow.context import RunContext
from tomolab.workflow.executor import PipelineExecutor
from tomolab.workflow.model import Pipeline, NodeInstance, Edge
from tomolab.workflow.registry import NodeRegistry
from tomolab.workflow.nodes.input_nodes import LoadProjectionsNode
from tomolab.workflow.nodes.preprocess_nodes import (
    ProjectionsToSinogramsNode,
    RingRemovalNode,
)

registry = NodeRegistry()
registry.register(LoadProjectionsNode)
registry.register(ProjectionsToSinogramsNode)
registry.register(RingRemovalNode)

pipeline = Pipeline(
    name="basic_pipeline",
    nodes=[
        NodeInstance(
            id="load1",
            type_name="load_projections",
            name="Load projections",
            params={"path": r"D:\tmp\small", "glob_pattern": "tomo_*.tif"},
        ),
        NodeInstance(
            id="sino1",
            type_name="projections_to_sinograms",
            name="Make sinograms",
            params={},
        ),
        NodeInstance(
            id="ring1",
            type_name="ring_removal",
            name="Ring removal",
            params={"correction": "algotom", "workers": 12},
        ),
    ],
    edges=[
        Edge("load1", "projections", "sino1", "projections"),
        Edge("sino1", "sinograms", "ring1", "sinograms"),
    ],
)

ctx = RunContext(
    project_dir=Path("."),
    run_dir=Path("D:/tests_CT/test_run"),
    temp_dir=Path("D:/tests_CT/test_run/tmp"),
    cache_dir=Path("D:/tests_CT/cache"),
)

ctx.run_dir.mkdir(parents=True, exist_ok=True)
ctx.temp_dir.mkdir(parents=True, exist_ok=True)
ctx.cache_dir.mkdir(parents=True, exist_ok=True)

executor = PipelineExecutor(registry)
results = executor.execute(pipeline, ctx)
print(results["ring1"].outputs["corrected_sinograms"].path)
