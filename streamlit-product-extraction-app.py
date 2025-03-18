import streamlit as st
import pandas as pd
import sqlite3
import tempfile
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import speech_recognition as sr
from dotenv import load_dotenv
from openpyxl import Workbook

# Import our extractors from modules
from api_models.deepseek_client import DeepSeekProductExtractor
from api_models.gemini_client import GeminiProductExtractor
from api_models.ollama_client import OllamaProductExtractor

load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Product & Price Extractor",
    page_icon="🛒",
    layout="wide"
)


# Database helpers


class DatabaseManager:
    """Handles all database operations for the product extractor app"""

    @staticmethod
    def adapt_datetime(dt: datetime) -> str:
        return dt.isoformat()

    @staticmethod
    def convert_datetime(bytestring: bytes) -> datetime:
        return datetime.fromisoformat(bytestring.decode())

    def __init__(self, db_path: str = 'product_extractions.db'):
        """Initialize database connection and setup tables"""
        self.db_path = db_path

        # Register adapters for datetime handling
        sqlite3.register_adapter(datetime, self.adapt_datetime)
        sqlite3.register_converter('DATETIME', self.convert_datetime)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist"""
        conn = sqlite3.connect(
            self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS products
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT NOT NULL,
         price REAL NOT NULL,
         currency TEXT NOT NULL,
         quantity INTEGER DEFAULT 1,
         source_text TEXT,
         model TEXT,
         extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        conn.commit()
        conn.close()

    def save_products(self, products: List[Dict[str, Any]]) -> int:
        """Save extracted products to the database"""
        if not products:
            return 0

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        count = 0
        for product in products:
            try:
                c.execute(
                    "INSERT INTO products (name, price, currency, quantity, source_text, model) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        product.get("name", "Unknown Product"),
                        float(product.get("price", 0.0)),
                        product.get("currency", "$"),
                        int(product.get("quantity", 1)),
                        product.get("source_text", ""),
                        product.get("model", "unknown")
                    )
                )
                count += 1
            except Exception as e:
                st.error(f"Error saving product to database: {e}")

        conn.commit()
        conn.close()
        return count

    def get_products(self, limit: int = 50) -> pd.DataFrame:
        """Retrieve products from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT id, name, price, currency, quantity, source_text, model, extracted_at FROM products ORDER BY extracted_at DESC LIMIT ?",
                conn,
                params=(limit,)
            )
            return df
        finally:
            conn.close()


# Speech Recognition
class SpeechProcessor:
    """Handles speech-to-text conversion"""

    @staticmethod
    def speech_to_text() -> Optional[str]:
        """Convert speech to text using Google's speech recognition API"""
        r = sr.Recognizer()

        # Adjust the recognizer sensitivity
        r.energy_threshold = 300  # Default is 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.0  # Wait 1 second of silence to consider the phrase complete

        with st.status("Listening...") as status:
            status.write("Please speak clearly. I'll listen until you pause.")
            with sr.Microphone() as source:
                # Add ambient noise adjustment
                r.adjust_for_ambient_noise(source, duration=1)
                st.info("🎤 Listening... Speak now")
                try:
                    # Increase timeout to allow for longer recordings
                    audio = r.listen(source, timeout=10, phrase_time_limit=15)
                    status.update(label="Processing speech...", state="running")
                    text = r.recognize_google(audio)
                    status.update(label="Done!", state="complete")
                    return text
                except sr.WaitTimeoutError:
                    st.error("No speech detected. Try again.")
                    return None
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please speak clearly.")
                    return None
                except sr.RequestError as e:
                    st.error(f"Speech service error: {e}")
                    return None


# App UI Components
class ProductExtractorApp:
    """Main application class for the Product Extractor App"""

    def __init__(self):
        """Initialize app components"""
        self.db_manager = DatabaseManager()
        self.speech_processor = SpeechProcessor()

    def _create_extractor(self, model_type: str, model_name: Optional[str] = None):
        extractors = {
            "Gemini": GeminiProductExtractor,
            "DeepSeek": DeepSeekProductExtractor,
            "Ollama": lambda: OllamaProductExtractor(model_name=model_name or "deepseek-r1:1.5b")
        }

        if model_type not in extractors:
            raise ValueError(f"Unknown model type: {model_type}")

        return extractors[model_type]() if model_type != "Ollama" else extractors[model_type]()

    def _display_products(self, products: List[Dict[str, Any]]) -> None:
        """Display extracted products in a dataframe"""
        if not products:
            st.warning("No products found in the text.")
            return

        st.success(f"Found {len(products)} products!")

        # Show results in a table
        display_products = [
            {k: v for k, v in p.items() if k not in ["source_text"]} for p in products]
        result_df = pd.DataFrame(display_products)
        st.dataframe(result_df, use_container_width=True)

        # Automatically save to database
        count = self.db_manager.save_products(products)
        st.success(f"Saved {count} products to database!")

        # Keep the manual save button for clarity (optional)
        # if st.button("Save to Database"):
        #     count = self.db_manager.save_products(products)
        #     st.success(f"Saved {count} products to database!")

    def _process_input(self, model_type: str, text: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process text input using selected model"""
        with st.spinner("Extracting products..."):
            try:
                extractor = self._create_extractor(model_type, model_name)
                return extractor.extract_products(text)
            except Exception as e:
                st.error(f"Error extracting products: {e}")
                return []

    def render_extract_tab(self) -> None:
        """Render the Extract tab"""
        st.subheader("Extract Products")

        # Model selection
        model_type = st.radio("Select AI model:", [
            "Gemini", "Ollama", "DeepSeek"], horizontal=True)

        model_name = None
        if model_type == "Ollama":
            model_name = st.selectbox(
                "Select Ollama model:",
                ["gemma3:4b", "deepseek-r1:1.5b"]
            )

        # Input method selection
        input_method = st.radio("Select input method:", [
            "Text", "Voice"], horizontal=True)

        if input_method == "Text":
            text_input = st.text_area(
                "Enter text containing products and prices:",
                height=100, placeholder="e.g., 'Apple iPhone 14 Pro Max $999, Samsung Galaxy S21 $799'"
            )

            if st.button("Extract Products") and text_input:
                products = self._process_input(
                    model_type, text_input, model_name)
                self._display_products(products)

        elif input_method == "Voice":
            if st.button("🎤 Record Voice"):
                text_input = self.speech_processor.speech_to_text()

                if text_input:
                    st.info(f"Transcribed text: {text_input}")
                    products = self._process_input(
                        model_type, text_input, model_name)
                    self._display_products(products)

    def render_database_tab(self) -> None:
        """Render the Database tab"""
        st.subheader("Database Records")

        # Initialize refresh state if not exists
        if 'refresh_db' not in st.session_state:
            st.session_state.refresh_db = False

        if st.button("Refresh Data"):
            st.session_state.refresh_db = True

        # Get products from database
        products_df = self.db_manager.get_products()

        if products_df.empty:
            st.info("No records found in the database yet. Extract some products first!")
            return

        if not products_df.empty:
            # Format the dataframe for display
            display_df = products_df.copy()
            display_df['formatted_price'] = display_df.apply(
                lambda row: f"{row['currency']} → {row['price']:.2f}", axis=1
            )

            # Just use # type: ignore comment to silence IDE warnings
            display_df = display_df[['id', 'name', 'formatted_price', 'quantity',
                                     'source_text', 'model', 'extracted_at']]  # type: ignore

            # Rename columns for display
            display_df.columns = ['ID', 'Product', 'Price', 'Quantity', 'Text', 'Model', 'Extracted At']

            st.dataframe(display_df, use_container_width=True)

            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export for Google Sheets"):
                    try:
                        csv = products_df.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="Download CSV for Google Sheets",
                            data=csv,
                            file_name=f"product_extractions_for_sheets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Import this CSV directly into Google Sheets"
                        )
                        st.success("CSV file ready! Click above to download.")
                    except Exception as e:
                        st.error(f"Error generating CSV file: {e}")
            with col2:
                if st.button("Export as Excel"):
                    try:
                        with st.spinner("Preparing Excel file..."):
                            buffer = io.BytesIO()

                            # Use pandas to_excel directly with ExcelWriter for more control
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                products_df.to_excel(writer, index=False, sheet_name='Products')
                                workbook = writer.book
                                worksheet = writer.sheets['Products']

                                # Auto-adjust column widths
                                for i, col in enumerate(products_df.columns):
                                    max_len = max(
                                        products_df[col].astype(str).map(len).max(),
                                        len(str(col))
                                    ) + 2
                                    worksheet.column_dimensions[chr(65 + i)].width = max_len

                            buffer.seek(0)

                            st.download_button(
                                label="Download Excel File",
                                data=buffer,
                                file_name=f"product_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    except Exception as e:
                        st.error(f"Error generating Excel file: {e}")

    def run(self) -> None:
        """Run the main application"""
        st.title("🛒 Product & Price Extractor")
        st.write("Extract product names and prices from text or voice input using AI")

        # Create tabs
        tab1, tab2 = st.tabs(["Extract", "Database"])

        with tab1:
            self.render_extract_tab()

        with tab2:
            self.render_database_tab()


# Main application entry point
if __name__ == "__main__":
    app = ProductExtractorApp()
    app.run()
