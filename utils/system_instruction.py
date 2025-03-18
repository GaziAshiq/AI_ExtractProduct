instruction = """
        Extract all products and their prices mentioned in the text.
        Return a JSON array where each item has the following format:
        {
            "name": "Product Name",
            "price": 123.45,
            "currency": "$",
            "quantity": 1
        }

        Rules:
        1. Extract complete product names including brand and model
        2. Convert all prices to numeric values (no currency symbols in the price field)
        3. Identify the currency symbol used ($, €, £, etc.) and include it separately
        4. Return an empty array if no products with prices are detected
        5. Do not make up any information not present in the text
        6. Extract bangla text as well
        7. If quantity is mentioned, include it, otherwise default to 1
        """
