import os
import time
import uuid
import logging
import tempfile
import shutil
from typing import List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import dotenv
from app import (
    load_document,
    split_documents,
    store_documents,
    initialize_weaviate_schema,
    weaviate_client,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("automated_pipeline.log"),
        logging.StreamHandler(),
    ],
)

# Load environment variables
dotenv.load_dotenv()

# Constants
WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "./documents_to_process")
PROCESSED_DIRECTORY = os.getenv("PROCESSED_DIRECTORY", "./processed_documents")
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".json"]
POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL", "5"))  # seconds


class DocumentHandler(FileSystemEventHandler):
    """Handler for file system events in the watched directory."""

    def __init__(self):
        super().__init__()
        # Initialize the Weaviate schema
        initialize_weaviate_schema()
        
        # Create directories if they don't exist
        os.makedirs(WATCH_DIRECTORY, exist_ok=True)
        os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
        
        logging.info(f"Watching directory: {WATCH_DIRECTORY}")
        logging.info(f"Processed files will be moved to: {PROCESSED_DIRECTORY}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() not in SUPPORTED_EXTENSIONS:
            logging.warning(f"Unsupported file type: {file_extension}. Skipping {file_path}")
            return
        
        logging.info(f"New file detected: {file_path}")
        self.process_file(file_path)

    def process_file(self, file_path):
        """Process a document file and store it in Weaviate."""
        try:
            # Get filename and extension
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(filename)[1].lower().lstrip('.')
            
            logging.info(f"Processing {filename}...")
            
            # Generate a deterministic doc_id based on filename
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
            
            # Load document
            documents = load_document(file_path, file_extension, filename)
            
            # Split into chunks
            chunks = split_documents(documents)
            
            # Store in Weaviate
            result = store_documents(chunks, doc_id)
            
            logging.info(f"Successfully processed {filename}: {result['message']}")
            
            # Move to processed directory
            processed_path = os.path.join(PROCESSED_DIRECTORY, filename)
            shutil.move(file_path, processed_path)
            logging.info(f"Moved {filename} to {PROCESSED_DIRECTORY}")
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")


def start_monitoring():
    """Start monitoring the watch directory for new files."""
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()
    
    logging.info("Started automated document processing pipeline")
    logging.info(f"Monitoring directory: {WATCH_DIRECTORY}")
    
    try:
        # Process any existing files in the directory
        for filename in os.listdir(WATCH_DIRECTORY):
            file_path = os.path.join(WATCH_DIRECTORY, filename)
            if os.path.isfile(file_path):
                _, file_extension = os.path.splitext(file_path)
                if file_extension.lower() in SUPPORTED_EXTENSIONS:
                    logging.info(f"Found existing file: {file_path}")
                    event_handler.process_file(file_path)
        
        # Keep the script running
        while True:
            time.sleep(POLLING_INTERVAL)
    
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping automated document processing pipeline")
    
    observer.join()


if __name__ == "__main__":
    start_monitoring()
