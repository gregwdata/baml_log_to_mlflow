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
    Extract from this content:
    {{ inventory_text }}

    {{ ctx.output_format }}
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
