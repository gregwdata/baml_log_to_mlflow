# 💥✍ baml_log_to_mlflow

Simple example for logging [BAML](https://docs.boundaryml.com/home) function calls to MLFlow traces.

Provides wrapper functions that use the BAML [Collector](https://docs.boundaryml.com/guide/baml-advanced/collector-track-tokens) to log raw LLM inputs/outputs and BAML results into the [MLFlow Trace schema](https://mlflow.org/docs/latest/tracing/)

>[!Warning]
>
> Currently only tested (and expected to work) with OpenAI and OpenAI-like API's. There are a few cases where chat messages are extracted from http requests and responses that assume that API schema. 

## Usage Examples

### Running in Codespaces

This repo is configured to run in GitHub Codespaces. 

The example is set up around using an OpenAI client. You will need to [provide a Codespaces environment variable](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces) called `OPENAI_API_KEY` with your key, or define that environment varible in your session. For other clients, you'll want to update the `baml_src` files accordingly, and provide the needed credentials.

Use the green button in the upper right to launch a Codespace. 

After initialization (may take a few minute the first time), it will have pre-installed the versions of `mlflow` and `baml-py` specified in `requirements.txt`.

In the terminal, run `mlflow ui`. Then click the button in the pop-up to open in browser to see the MLFlow interface.

In a second terminal, run `python demo_module_import.py`.

Now, when you refresh your MLFlow window, you can see the logged traces by clicking on the experiment name, then the Traces tab.

### Tracing a single BAML function

The `trace_baml_function` method wraps the BAML client function and passes arguments (as positional arguments in order following the function and/or keyword arguments) to it. 

```python
from trace_baml_function import trace_baml_function
from baml_client import b

result = trace_baml_function(b.<FunctionName>, <function arguments>)
```

For example, the single-function example in `demo_module_import.py` is

```python
from trace_baml_function import trace_baml_function
from baml_client import b

inventory_text: str = '''
Current Stock:
- Apples: 100 units, $0.50 each, SKU: APL123
- Oranges: 75 units, $0.75 each, SKU: ORG456
- Bananas: 50 units, $0.60 each, SKU: BAN789
'''

items_single = trace_baml_function(b.ListInventory, inventory_text)
print("Single-call trace items:")
for item in items_single:
    print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")
```

It produces the following output in the terminal:

```
Single-call trace items:
Apples: 100 units at $0.5 each (SKU: APL123)
Oranges: 75 units at $0.75 each (SKU: ORG456)
Bananas: 50 units at $0.6 each (SKU: BAN789)
```

And the MLFlow Trace will look like:

![Screen capture of an MLflow trace interface for the experiment "baml_trace." The left pane shows the trace structure with a task named "ListInventory" and a subtask "LLMCall," each with execution times near 1 second. The main panel is on the "Chat" tab, showing a prompt-response interaction. The system prompt instructs the assistant to extract inventory data in a specified JSON schema. The user lists the current stock of apples, oranges, and bananas. The assistant responds with a correctly formatted JSON array capturing item name, quantity, price, and SKU for each product.](readme_images/single_call_trace.png)

By default, when called this way, the Experiment name will be set from the name of the BAML function. In this case, `ListInventory`.

>[!Note]
>To specify a custom experiment name, you can use the context manager method described in the following section, even for a single BAML function.


### Tracing multiple BAML functions

To create a parent trace, within which multiple BAML functions can have their own spans, the `start_baml_trace` method should be used as a context manager. It's input argument will be the MLFlow experiment name. It returns a `req_id`, which identifies the trace, and a `root_id` which identifies the root span of the trace. These should be passed on to the individual `trace_baml_function` calls to attach them to this trace. If desired, additional `mlflow.start_span` context managers can be used to further structure the span hierarchy.

```python
from trace_baml_function import trace_baml_function
from baml_client import b

with start_baml_trace("experiment name") as (req_id, root_id):
    result1 = trace_baml_function(
        b.Function1,
        input1,
        keyword1=keyword_input1,
        request_id=req_id,
        parent_id=root_id
    )
    result2 = trace_baml_function(
        b.Function2,
        input2,
        keyword1=keyword_input2,
        request_id=req_id,
        parent_id=root_id
    )
```

The example for applying this in `demo_module_import.py` uses two functions. One to record the inventory, and one to update it based on a natural-language user input:

```python
from trace_baml_function import start_baml_trace, trace_baml_function
from baml_client import b

inventory_text: str = '''
Current Stock:
- Apples: 100 units, $0.50 each, SKU: APL123
- Oranges: 75 units, $0.75 each, SKU: ORG456
- Bananas: 50 units, $0.60 each, SKU: BAN789
'''
update_message: str = "I just received a shipment of 20 apples, and sold 5 oranges."

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
```

It's printed output is:
```
Multi-call trace initial items:
Apples: 100 units at $0.5 each (SKU: APL123)
Oranges: 75 units at $0.75 each (SKU: ORG456)
Bananas: 50 units at $0.6 each (SKU: BAN789)
Multi-call trace updated items:
Apples: 120 units at $0.5 each (SKU: APL123)
Oranges: 70 units at $0.75 each (SKU: ORG456)
Bananas: 50 units at $0.6 each (SKU: BAN789)
```

![Screenshot of an MLflow UI showing a workflow named "baml_multi_workflow." The workflow consists of tasks: ListInventory, UpdateInventory, and associated LLM calls (LLMCallopenai_1, LLMCallopenai_2). On the right panel under the "Chat" tab, a system message provides instructions for updating inventory using JSON format. It includes a sample inventory with items (Apples, Oranges, Bananas) and a user message: "I just received a shipment of 20 apples, and sold 5 oranges." The assistant responds with updated inventory JSON reflecting changes: apples increased to 120, oranges decreased to 70, and bananas unchanged. Task durations and execution order are shown in a Gantt-like bar chart.](readme_images/multi_call_trace.png)