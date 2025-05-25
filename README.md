# 💥✍ baml_log_to_mlflow

Simple example for logging [BAML](https://docs.boundaryml.com/home) function calls to MLFlow traces.

Provides wrapper functions that use the BAML [Collector](https://docs.boundaryml.com/guide/baml-advanced/collector-track-tokens) to log raw LLM inputs/outputs and BAML results into the [MLFlow Trace schema](https://mlflow.org/docs/latest/tracing/)

>[!Warning]
>
> Currently only tested (and expected to work) with OpenAI and OpenAI-like API's. There are a few cases where chat messages are extracted from http requests and responses that assume that API schema. 

## Usage Examples

