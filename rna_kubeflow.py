import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics
)


@component(
    packages_to_install=["pandas", "numpy"],
    output_component_file="preprocess_data_component.yaml"
)
def preprocess_train_data(input_x_filename:str, X_output_csv:Output[Dataset]):
    import numpy as np
    import pandas as pd

    # load training data
    X_train = pd.read_csv(input_x_filename, index_col=0)

    # create correlation matrix
    corr_matrix = X_train.corr().abs()

    # select upper (or lower) triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # find features with correlation greater than 0.95
    correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

    # drop correlated features
    X_train.drop(correlated_features, axis=1, inplace=True)

    # output updated data to csv
    X_train.to_csv(X_output_csv.path)


@component(
    packages_to_install=["pandas", "scikit-learn"],
    output_component_file="split_data_component.yaml"
)
def split_data(input_x:Input[Dataset], input_y:str, x_train:Output[Dataset], x_test:Output[Dataset], y_train:Output[Dataset], y_test:Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X_data = pd.read_csv(input_x.path)
    y_data = pd.read_csv(input_y)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

    X_train.to_csv(x_train.path)
    X_test.to_csv(x_test.path)
    y_train.to_csv(y_train.path)
    y_test.to_csv(y_test.path)


@dsl.pipeline(
    name="rna-seq-pipeline",
)
def rnaseq_pipeline(x_data:str, y_data:str):
    preprocess_data_task = preprocess_train_data(x_data)
    split_data_task = split_data(preprocess_data_task.outputs['X_output_csv'], y_data)


kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
pipeline_func=rnaseq_pipeline,
package_path='pipeline.yaml')