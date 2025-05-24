// Defining a data model.
class Inventory {
  item string @description(#"The name of the item."#)
  quantity int @description(#"The quantity of the item."#)
  price float @description(#"The price of the item."#)
  sku string @description(#"The SKU identifier of the item."#)
}

// Create a function to extract inventory items from a text.
function ListInventory(inventory_text: string) -> Inventory[] {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Extract the inventory items from the following content. 

    {{ ctx.output_format }}

    {{ _.role("user") }} {{ inventory_text }}
  "#
}

// Function to update the inventory based on free text input
function UpdateInventory(
  current_inventory: Inventory[],
  change_message: string
) -> Inventory[] {
  client "openai/gpt-4o"
  prompt #"
    Given the current inventory and a message about changes, update the inventory accordingly. 
    Make sure to:
    - Keep all existing items that weren't modified
    - Only modify quantities for mentioned items
    - Ensure quantities don't go below 0
    - Preserve original SKUs and prices

    Current inventory:
    {{ current_inventory }}

    {{ ctx.output_format }}

    {{ _.role("user") }} {{ change_message }}
  "#
}



// Test the function with a sample input. Open the VSCode playground to run this.
test test_list_inventory {
  functions [ListInventory]
  args {
    inventory_text #"
      Inventory:
      - item: Apple
        quantity: 10
        price: 0.5
        sku: A123
      - item: Banana
        quantity: 20
        price: 0.3
        sku: B456
    "#
  }
}


test TestUpdateInventory {
  functions [UpdateInventory]
  args {
    current_inventory [
      {
        item "Apple"
        quantity 50
        price 1.25
        sku "APL001"
      }
      {
        item "Banana"
        quantity 30
        price 0.75
        sku "BAN002"
      }
    ]
    change_message "I just sold 5 apples and received a shipment of 10 more bananas"
  }
}

test TestUpdateInventoryNegative {
  functions [UpdateInventory]
  args {
    current_inventory [
      {
        item "Apple"
        quantity 3
        price 1.25
        sku "APL001"
      }
    ]
    change_message "I sold 5 apples"
  }
}
