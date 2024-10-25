import logging
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from src.agent import VerificationPipeline
from config.settings import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable proxy settings if active
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

def main():
    # Streamlit UI for user input
    st.title("Sentiment and Narrative Verification")
    input_text = st.text_area("Enter the text for verification:")

    if st.button("Run Verification"):
        # Initialize OpenAI LLMs using the provided API key
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not found. Ensure that your .env file is correctly set up.")
            return

        try:
            # Load models using OpenAI API key
            logger.info("Initializing language models...")
            llm_35 = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0, 
                openai_api_key=OPENAI_API_KEY
            )
            llm_4 = ChatOpenAI(
                model_name="gpt-4", 
                temperature=0, 
                openai_api_key=OPENAI_API_KEY
            )

            # Log the input
            logger.info(f"Input for processing: {input_text}")

            # Initialize the verification pipeline
            verification_process = VerificationPipeline(llm_35, llm_4, input_text)

            # Run the pipeline and capture the result
            logger.info("Running the verification process...")
            result = verification_process.process_news()

            # Display the output
            if result:
                logger.info("Verification process completed. Displaying result:")
                st.write(result)
            else:
                logger.warning("Verification process did not return any result.")
                st.warning("Verification process did not return any result.")

        except Exception as e:
            logger.error(f"An error occurred during the verification process: {e}", exc_info=True)
            st.error(f"An error occurred during the verification process: {e}")

if __name__ == "__main__":
    main()
