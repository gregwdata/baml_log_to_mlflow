'''
Module for tracing BAML client function calls and logging them to MLflow,
with automatic standalone trace creation and multi-call trace grouping via start_baml_trace.
'''

import traceback
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Iterator, Callable

from baml_client import b
from baml_py import Collector
from baml_py.errors import BamlError
from mlflow import MlflowClient, set_experiment
from mlflow.entities.span import SpanType
from mlflow.tracing import set_span_chat_messages, set_span_chat_tools

@contextmanager
def start_baml_trace(experiment: str) -> Iterator[Tuple[str, str]]:
    """
    Context manager for grouping multiple BAML calls into a single MLflow trace.

    Args:
        experiment (str): Name of the MLflow experiment.

    Yields:
        Tuple[str, str]: A tuple of (request_id, root_span_id) for the active trace.
    """
    set_experiment(experiment)
    client = MlflowClient()
    root_span = client.start_trace(
        name="baml_multi_workflow",
        tags={"experiment": experiment},
        span_type=SpanType.CHAIN,
        inputs={}
    )
    request_id: str = root_span.request_id
    root_span_id: str = root_span.span_id
    try:
        yield request_id, root_span_id
    finally:
        client.end_trace(request_id=request_id)


def trace_baml_function(
    func: Callable[..., Any],
    *args: Any,
    request_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    Trace a BAML function call and log its activity to MLflow.

    If request_id is None, starts a standalone trace using the function's name.
    If parent_id is None, fetches the root span ID for the given request_id.

    Args:
        func (Callable[..., Any]): The BAML client function to invoke.
        *args: Positional arguments passed to func.
        request_id (Optional[str]): MLflow trace request_id. Defaults to None.
        parent_id (Optional[str]): Parent span ID for new spans. Defaults to None.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Any: Result of the BAML function, or error info if a BAML error occurs.
    """
    # Start standalone trace if no request_id
    if request_id is None:
        with start_baml_trace(func.__name__) as (rid, pid):
            return trace_baml_function(
                func, *args, request_id=rid, parent_id=pid, **kwargs
            )

    client = MlflowClient()
    # Lookup root span if parent_id not provided
    if parent_id is None:
        trace_obj = client.get_trace(request_id)
        parent_id = trace_obj.root_span_id

    try:
        collector = Collector(name="baml_collector")
        result: Any = func(*args, baml_options={"collector": collector}, **kwargs)

        for log in collector.logs:
            func_span = client.start_span(
                name=log.function_name,
                request_id=request_id,
                parent_id=parent_id,
                span_type=SpanType.AGENT,
                inputs={'args': args, 'kwargs': kwargs},
                attributes={
                    "baml.id": log.id,
                    "baml.tokens_in": log.usage.input_tokens,
                    "baml.tokens_out": log.usage.output_tokens
                },
                start_time_ns=log.timing.start_time_utc_ms * 1_000_000
            )

            for call in log.calls:
                cs = client.start_span(
                    name=f"LLMCall:{call.provider}",
                    request_id=request_id,
                    parent_id=func_span.span_id,
                    inputs=call.http_request.body.json(),
                    attributes={
                        "status": call.http_response.status,
                        "tokens_in": call.usage.input_tokens,
                        "tokens_out": call.usage.output_tokens
                    },
                    start_time_ns=call.timing.start_time_utc_ms * 1_000_000,
                    span_type=SpanType.CHAT_MODEL
                )
                set_span_chat_messages(
                    cs,
                    call.http_request.body.json()["messages"] +
                    [call.http_response.body.json()["choices"][0]["message"]]
                )
                # Optionally attach tools: set_span_chat_tools(cs, call.chat_tools)

                client.end_span(
                    request_id=request_id,
                    span_id=cs.span_id,
                    outputs={"resp": call.http_response.body.json()},
                    end_time_ns=(call.timing.start_time_utc_ms + call.timing.duration_ms) * 1_000_000
                )

            client.end_span(
                request_id=request_id,
                span_id=func_span.span_id,
                outputs={"out": log.raw_llm_response},
                end_time_ns=(log.timing.start_time_utc_ms + log.timing.duration_ms) * 1_000_000
            )
    except BamlError as e:
        print(f"BAML Error: {e}")
        result = [{"error": str(e), "traceback": traceback.format_exc()}]
    return result


def main() -> None:
    """
    Demonstrate standalone and multi-call tracing for BAML functions.
    """
    inventory_text: str = '''
    Current Stock:
    - Apples: 100 units, $0.50 each, SKU: APL123
    - Oranges: 75 units, $0.75 each, SKU: ORG456
    - Bananas: 50 units, $0.60 each, SKU: BAN789
    '''
    update_message: str = "I just received a shipment of 20 apples, and sold 5 oranges."

    # Standalone trace example
    items_single = trace_baml_function(b.ListInventory, inventory_text)
    print("Single-call trace items:")
    for item in items_single:
        print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")

    # Multi-call trace example
    with start_baml_trace("baml_inventory_multi") as (req_id, root_id):
        items1 = trace_baml_function(
            b.ListInventory,
            inventory_text,
            request_id=req_id,
            parent_id=root_id
        )
        items2 = trace_baml_function(
            b.UpdateInventory,
            items1, # Use the items from the first call
            update_message,
            request_id=req_id,
            parent_id=root_id
        )

    print("Multi-call trace initial items:")
    for item in items1:
        print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")

    print("Multi-call trace updated items:")
    for item in items2:
        print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")


if __name__ == '__main__':
    main()
