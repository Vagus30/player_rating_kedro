"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict_next_values_bowl,predict_next_values_bat


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict_next_values_bowl,
                inputs=["test_bowl"],
                outputs="prediction_bowl",
                name="predict_next_values_bowl_node",
            ),
            node(
                func=predict_next_values_bat,
                inputs=["test_bat"],
                outputs="prediction_bat",
                name="predict_next_values_bat_node",
            ),            

        ]
    )
