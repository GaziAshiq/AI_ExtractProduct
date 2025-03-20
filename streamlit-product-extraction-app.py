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
from indic_transliteration import sanscript

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
        """Create tables if they don't exist and update schema if needed"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()

        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
        table_exists = c.fetchone()

        if not table_exists:
            # Create new table with all columns
            c.execute('''
            CREATE TABLE products
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT NOT NULL,
             price REAL NOT NULL,
             currency TEXT NOT NULL,
             quantity INTEGER DEFAULT 1,
             quantity_description TEXT,
             quantity_multiplier REAL DEFAULT 1.0,
             source_text TEXT,
             model TEXT,
             extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
            ''')
        else:
            # Check if model column exists
            c.execute("PRAGMA table_info(products)")
            # columns = [col[1] for col in c.fetchall()]
            #
            # # Add missing columns if needed
            # if 'model' not in columns:
            #     c.execute("ALTER TABLE products ADD COLUMN model TEXT")

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
                    "INSERT INTO products (name, price, currency, quantity, quantity_description, quantity_multiplier, source_text, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        product.get("name", "Unknown Product"),
                        float(product.get("price", 0.0)),
                        product.get("currency", "$"),
                        int(product.get("quantity", 1)),
                        product.get("quantity_description", ""),
                        float(product.get("quantity_multiplier", 1.0)),
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
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)

            # Get column names first to handle missing columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(products)")
            columns = [col[1] for col in cursor.fetchall()]

            # Build query based on available columns
            select_columns = []
            database_columns = ["id", "name", "price", "currency", "quantity", "quantity_description",
                                "quantity_multiplier",
                                "source_text", "model", "extracted_at"]

            for col in database_columns:
                if col in columns:
                    select_columns.append(col)

            query = f"SELECT {', '.join(select_columns)} FROM products ORDER BY extracted_at DESC LIMIT ?"

            df = pd.read_sql_query(query, conn, params=(limit,))

            # Add any missing columns with default values
            for col in database_columns:
                if col not in df.columns:
                    if col == "model":
                        df[col] = "unknown"
                    elif col == "extracted_at":
                        df[col] = datetime.now()
                    else:
                        df[col] = ""

            return df
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()


# Speech Recognition
class SpeechProcessor:
    """Handles speech-to-text conversion"""

    @staticmethod
    def speech_to_text(language: str = "bn-BD") -> Optional[str]:
        """Convert speech to text using Google's speech recognition API
        Args:
            language: Language code (bn-BD for Bengali, en-US for English)
        Returns:
            Extracted text in the specified language format, or None if an error occurs
        """
        r = sr.Recognizer()

        # Adjust the recognizer sensitivity
        r.energy_threshold = 300  # Higher threshold for less noise
        r.dynamic_energy_threshold = True
        r.pause_threshold = 2.0  # Longer pause threshold (seconds)
        r.phrase_threshold = 0.5  # Minimum length of silence to consider end of phrase
        r.non_speaking_duration = 1.0  # Additional time after speech ends to ensure completeness

        # Determine language-specific messages
        messages = {
            "bn-BD": {
                "listen_msg": "অনুগ্রহ করে স্পষ্টভাবে কথা বলুন। শেষ হলে কিছুক্ষণ থামুন।",
                "adjusting": "🎤 পার্শ্ববর্তী শব্দের জন্য সামঞ্জস্য করা হচ্ছে...",
                "recording": "🎤 রেকর্ডিং চলছে... কথা বলুন এবং শেষ হলে থামুন",
                "processing": "কথা প্রক্রিয়াকরণ হচ্ছে...",
                "done": "সম্পন্ন হয়েছে!",
                "no_speech": "কোন কথা শনাক্ত হয়নি। আবার চেষ্টা করুন।",
                "not_understand": "কথা বোঝা যায়নি। স্পষ্টভাবে কথা বলুন।",
                "service_error": "স্পিচ সার্ভিস ত্রুটি:",
                "mic_error": "মাইক্রোফোন সমস্যা:"
            },
            "en-US": {
                "listen_msg": "Please speak clearly. Pause when you're finished.",
                "adjusting": "🎤 Adjusting for background noise...",
                "recording": "🎤 Recording... speak now and pause when done",
                "processing": "Processing speech...",
                "done": "Done!",
                "no_speech": "No speech detected. Try again.",
                "not_understand": "Could not understand audio. Please speak clearly.",
                "service_error": "Speech service error:",
                "mic_error": "Microphone error:"
            }
        }
        # Get messages for selected language, default to English if not found
        msg = messages.get(language, messages["en-US"])

        try:
            with st.status("Listening...", expanded=True) as status:
                status.write(msg["listen_msg"])

                with sr.Microphone() as source:
                    st.info(msg["adjusting"])
                    # Adjust for ambient noise with longer duration for better results
                    r.adjust_for_ambient_noise(source, duration=2)
                    st.info(msg["recording"])

                    try:
                        # Increased timeout and phrase_time_limit for longer speech
                        audio = r.listen(source, timeout=20, phrase_time_limit=60)
                        status.update(label=msg["processing"], state="running")

                        # Use specified language model for recognition
                        text = r.recognize_google(audio, language=language)
                        status.update(label=msg["done"], state="complete")
                        return text

                    except sr.WaitTimeoutError:
                        st.error(msg["no_speech"])
                    except sr.UnknownValueError:
                        st.error(msg["not_understand"])
                    except sr.RequestError as e:
                        st.error(f"{msg['service_error']} {e}")

        except Exception as e:
            st.error(f"Error: {e}")

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
            language_option = st.selectbox(
                "Select language for voice input:",
                [
                    "bn-BD",  # Bangla
                    "en-US"  # English
                ],
                format_func=lambda x: "Bangla" if x == "bn-BD" else "English"
            )
            if st.button("🎤 Record Voice"):
                text_input = self.speech_processor.speech_to_text(language=language_option)

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

        # Format the dataframe for display
        display_df = products_df.copy()

        # Add formatted price column
        if 'currency' in display_df.columns and 'price' in display_df.columns:
            display_df['formatted_price'] = display_df.apply(
                lambda row: f"{row['currency']} → {row['price']:.2f}", axis=1
            )

        # Select only columns that exist in the dataframe
        columns_to_display = ["id", "name", "price", "currency", "quantity", "quantity_description",
                              "quantity_multiplier",
                              "source_text", "model", "extracted_at"]
        # existing_columns = [col for col in columns_to_display if col in display_df.columns or col == 'formatted_price']

        # display_df = display_df[existing_columns]
        display_df = display_df[columns_to_display]

        # Don't rename columns - just use what we have
        st.dataframe(display_df, use_container_width=True)

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as CSV"):
                try:
                    csv = products_df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="Download CSV for Google Sheets",
                        data=csv,
                        file_name=f"product_extractions{datetime.now().strftime('%Y%m%d')}.csv",
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
                            # file_name=f"product_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            file_name=f"product_extractions_{datetime.now().strftime('%Y%m%d')}.xlsx",
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


if __name__ == "__main__":
    app = ProductExtractorApp()
    app.run()
