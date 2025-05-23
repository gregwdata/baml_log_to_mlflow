from baml_client import b
from baml_client.types import Inventory

def main():
    # Sample inventory text
    inventory_text = '''
    Current Stock:
    - Apples: 100 units, $0.50 each, SKU: APL123
    - Oranges: 75 units, $0.75 each, SKU: ORG456
    - Bananas: 50 units, $0.60 each, SKU: BAN789
    '''
    
    # Extract inventory items
    inventory_items = b.ListInventory(inventory_text)
        
    # Print results
    print("Inventory Items:")
    for item in inventory_items:
        print(f"{item.item}: {item.quantity} units at ${item.price} each (SKU: {item.sku})")

if __name__ == '__main__':
    main()