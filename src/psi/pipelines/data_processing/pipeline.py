"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import data_processing,do_pca,change_to_dict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_processing,
                inputs="formatted_bbb_df",
                outputs=["merged_df","batsman_df_1","bowler_df_1"],
                name="data_processing_node",
            ),
            node(
                func=do_pca,
                inputs="merged_df",
                outputs="processed_merged_df",
                name="do_pca_node",
            ),
            node(
                func=change_to_dict,
                inputs=["bowler_df_1","batsman_df_1"],
                outputs=["test_bowl","test_bat"],
                name="change_to_dict_node",
            ),
        ]
    )
