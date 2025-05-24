from baml_client import b

from baml_py import Collector
from baml_py.errors import BamlError
from mlflow import MlflowClient, set_experiment
from mlflow.entities.span import SpanType
from mlflow.tracing import set_span_chat_messages, set_span_chat_tools
import traceback

def trace_baml_function(experiment, func, *args, **kwargs):
    """
    Run a BAML function with the given arguments and keyword arguments.
    """
    
    set_experiment(experiment)
    client = MlflowClient()
    root_span = client.start_trace("baml_workflow",
                                   tags={"function": func.__name__},
                                   span_type=SpanType.CHAIN,
                                   inputs={"args": args, "kwargs": kwargs}
                                   )
    req_id = root_span.request_id
    try: 
        # Initialize the collector
        collector = Collector(name="baml_collector")

        # Call the BAML function
        result = func(*args, baml_options={"collector": collector}, **kwargs)

        logs = collector.logs

        # 3. Map logs to MLflow spans
        for log in collector.logs:
            print(f"Log ID: {log.id}")
            print(f"Function Name: {log.function_name}")
            print(f"Start Time: {log.timing.start_time_utc_ms}")
            print(log)

            func_span = client.start_span(
                name=log.function_name, request_id=req_id, parent_id=root_span.span_id,
                span_type=SpanType.AGENT,
                attributes={
                    "baml.id": log.id, "baml.tokens_in": log.usage.input_tokens,
                    "baml.tokens_out": log.usage.output_tokens
                },
                start_time_ns=log.timing.start_time_utc_ms * 1_000_000
            )
            for call in log.calls:
                cs = client.start_span(
                    name=f"LLMCall:{call.provider}", request_id=req_id,
                    parent_id=func_span.span_id, inputs=call.http_request.body.json(),
                    attributes={
                        "status": call.http_response.status,
                        "tokens_in": call.usage.input_tokens,
                        "tokens_out": call.usage.output_tokens
                    },
                    start_time_ns=call.timing.start_time_utc_ms * 1_000_000,
                    span_type=SpanType.CHAT_MODEL
                )
                # Set the chat messages and tools for the span
                set_span_chat_messages(cs, call.http_request.body.json()['messages'] + [call.http_response.body.json()['choices'][0]['message']])
                # set_span_chat_tools(cs, call.chat_tools) # no tool calling in BAML
                client.end_span(request_id=req_id, span_id=cs.span_id, outputs={"resp": call.http_response.body.json()},
                                end_time_ns=(call.timing.start_time_utc_ms + call.timing.duration_ms) * 1_000_000,)
            client.end_span(request_id=req_id, span_id=func_span.span_id, outputs={"out": log.raw_llm_response},
                            end_time_ns=(log.timing.start_time_utc_ms + log.timing.duration_ms) * 1_000_000,)
    except BamlError as e:
        # Handle errors from the BAML call
        print(f"BAML Error: {e}")        
        result = [{
            "error": str(e),
            "traceback": traceback.format_exc()
        }]
    finally: # ensure we always close the trace and child spans
        client.end_trace(
            request_id=req_id,
            outputs={
                "total_input_tokens": collector.usage.input_tokens,
                "total_output_tokens": collector.usage.output_tokens,
                "result":result
            }
        )

    # Return the result and logs
    return result

def main():
    # Sample inventory text
    inventory_text = '''
    Current Stock:
    - Apples: 100 units, $0.50 each, SKU: APL123
    - Oranges: 75 units, $0.75 each, SKU: ORG456
    - Bananas: 50 units, $0.60 each, SKU: BAN789
    '''

    # Extract inventory items
    inventory_items = trace_baml_function("baml_inventory",b.ListInventory,inventory_text)

    # Print results
    print("Inventory Items:")
    for item in inventory_items:
        print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")


if __name__ == '__main__':
    main()