from trace_baml_function import start_baml_trace, trace_baml_function
from baml_client import b

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