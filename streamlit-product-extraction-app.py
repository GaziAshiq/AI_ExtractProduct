import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import tempfile
import json
import time
from datetime import datetime
from google import genai
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Product & Price Extractor",
    page_icon="🛒",
    layout="wide"
)


# Setup database
def init_db():
    conn = sqlite3.connect('product_extractions.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS products
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     name TEXT NOT NULL,
     price REAL NOT NULL,
     currency TEXT NOT NULL,
     quantity INTEGER DEFAULT 1,
     source_text TEXT,
     extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()


# Initialize database
init_db()


# Configure Gemini AI
@st.cache_resource
def setup_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"  # Next-gen features, speed, and multimodal generation
    return model


# Extract products from text
def extract_products(text, model):
    system_prompt = """
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
    4. If quantity is mentioned, include it, otherwise default to 1
    5. Return an empty array if no products with prices are detected
    6. Do not make up any information not present in the text
    """

    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system_prompt}\n\nText to extract from:\n{text}")

        # Try to parse JSON from the response
        try:
            # First, try direct parsing
            result = json.loads(response.text)
            if isinstance(result, dict) and "products" in result:
                # Some models might wrap in a 'products' key
                return result["products"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except json.JSONDecodeError:
            # If that fails, try to find JSON in the response text
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response.text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass

            # If all else fails, return empty list
            st.error(f"Failed to parse JSON from model response. Raw response: {response.text[:200]}...")
            return []

    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return []


# Save products to a database
def save_to_database(products, source_text):
    if not products:
        return 0

    conn = sqlite3.connect('product_extractions.db')
    c = conn.cursor()

    count = 0
    for product in products:
        try:
            c.execute(
                "INSERT INTO products (name, price, currency, quantity, source_text) VALUES (?, ?, ?, ?, ?)",
                (
                    product.get("name", "Unknown Product"),
                    float(product.get("price", 0.0)),
                    product.get("currency", "$"),
                    int(product.get("quantity", 1)),
                    source_text
                )
            )
            count += 1
        except Exception as e:
            st.error(f"Error saving product to database: {e}")

    conn.commit()
    conn.close()
    return count


# Get products from database
def get_products_from_db(limit=50):
    conn = sqlite3.connect('product_extractions.db')
    df = pd.read_sql_query(
        "SELECT id, name, price, currency, quantity, source_text, extracted_at FROM products ORDER BY extracted_at DESC LIMIT ?",
        conn,
        params=(limit,)
    )
    conn.close()
    return df


# Convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source)
        st.info("Processing speech...")

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Error with speech recognition service: {e}")
        return None


# Main application
def main():
    st.title("🛒 Product & Price Extractor")
    st.write("Extract product names and prices from text or voice input using Google's Gemini AI")

    # Initialize Gemini model
    model = setup_genai()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Extract", "Database", "About"])

    with tab1:
        st.subheader("Extract Products")

        # Input method selection
        input_method = st.radio("Select input method:", ["Text", "Voice"], horizontal=True)

        if input_method == "Text":
            text_input = st.text_area("Enter text containing products and prices:",
                                      "I bought a MacBook Pro for $1299 and an iPhone 14 for $799. Also grabbed some AirPods Pro for $249.",
                                      height=150)
            process = st.button("Extract Products")

            if process and text_input:
                with st.spinner("Extracting products..."):
                    products = extract_products(text_input, model)

                if products:
                    st.success(f"Found {len(products)} products!")

                    # Show results in a table
                    result_df = pd.DataFrame(products)
                    st.dataframe(result_df, use_container_width=True)

                    # Save to database
                    if st.button("Save to Database"):
                        count = save_to_database(products, text_input)
                        st.success(f"Saved {count} products to database!")
                else:
                    st.warning("No products found in the text.")

        elif input_method == "Voice":
            if st.button("🎤 Record Voice"):
                text_input = speech_to_text()

                if text_input:
                    st.info(f"Transcribed text: {text_input}")

                    with st.spinner("Extracting products..."):
                        products = extract_products(text_input, model)

                    if products:
                        st.success(f"Found {len(products)} products!")

                        # Show results in a table
                        result_df = pd.DataFrame(products)
                        st.dataframe(result_df, use_container_width=True)

                        # Save to database
                        if st.button("Save to Database"):
                            count = save_to_database(products, text_input)
                            st.success(f"Saved {count} products to database!")
                    else:
                        st.warning("No products found in the transcribed text.")

    with tab2:
        st.subheader("Database Records")

        if st.button("Refresh Data"):
            st.session_state.refresh_db = True

        # Get products from database
        products_df = get_products_from_db()

        if not products_df.empty:
            # Format the dataframe for display
            display_df = products_df.copy()
            display_df['formatted_price'] = display_df.apply(
                lambda row: f"{row['currency']}{row['price']:.2f}", axis=1
            )
            display_df = display_df[['id', 'name', 'formatted_price', 'quantity', 'extracted_at']]
            display_df.columns = ['ID', 'Product', 'Price', 'Quantity', 'Extracted At']

            st.dataframe(display_df, use_container_width=True)

            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV"):
                    csv = products_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"product_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            with col2:
                if st.button("Export as Excel"):
                    with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp:
                        products_df.to_excel(tmp.name, index=False)
                        with open(tmp.name, "rb") as f:
                            excel_data = f.read()
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name=f"product_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        else:
            st.info("No products found in the database. Extract some products first!")

    with tab3:
        st.subheader("About This App")
        st.write("""
        This application uses Google's Gemini AI to extract product names and prices from text or voice input.
        
        ### Features:
        - Extract products and prices from text input
        - Voice input support for hands-free extraction
        - Database storage of extracted products
        - Export functionality (CSV, Excel)
        
        ### How it works:
        1. Input text containing product information or use voice recording
        2. Gemini AI analyzes the text to identify products and prices
        3. Results are displayed and can be saved to a local SQLite database
        4. View and export your product database any time
        
        ### Technical Details:
        - Built with Streamlit
        - Powered by Google's Generative AI (Gemini)
        - Speech recognition using SpeechRecognition library
        - SQLite database for local storage
        """)


if __name__ == "__main__":
    main()
