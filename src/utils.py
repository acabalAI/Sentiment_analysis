import json
import requests
from urllib.parse import urlparse
import logging

# Define a dictionary of reliable sources with their respective reliability scores
RELIABLE_SOURCES = {
    "accion-ciudadana.org": 5.00,
    "tracoda.info": 5.00,
    "elconfidencial.com": 4.67,
    "elpais.com": 4.67,
    "eldiario.es": 4.33,
    "abc.es": 4.00,
    "europapress.es": 4.00,
    "lavanguardia.com": 4.33,
    "20minutos.es": 4.00,
    "elespanol.com": 4.00,
    "huffingtonpost.es": 4.00,
    "publico.es": 4.00,
    "larazon.es": 4.00,
    "eldiarioar.com": 4.33,
    "cnnespanol.cnn.com": 4.00,
    "bbc.com/mundo": 4.00,
    "rtve.es": 4.33,
    "efe.com": 4.00,
    "clarin.com": 4.00,
    "infobae.com": 4.00,
    "pagina12.com.ar": 4.00,
    "lavozdegalicia.es": 4.00,
    "laopinion.com": 4.00,
    "elespectador.com": 4.00,
    "semana.com": 4.00,
    "elmundo.es": 4.00,
    "nacion.com": 4.00,
    "elpais.com.uy": 4.00,
    "eltiempo.com": 4.00,
    "elpais.com.mx": 4.00,
    "eluniversal.com.mx": 4.00,
    "laopiniondemurcia.es": 4.00,
    "es.wikipedia.org": 5.00,
    "bloomberglinea.com": 1.00,
    "latimes.com": 1.00
}

# Extract the base domain from a URL
def extract_base_domain(url):
    """
    Extract the base domain from a given URL.

    Args:
        url (str): The URL to extract the domain from.

    Returns:
        str: The base domain of the URL.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

# Extract information using the Serper API
def info_extraction(subject, length=500, api_key=None, min_search=30):
    """
    Extract information related to a given subject using the Serper API.

    Args:
        subject (str): The subject to search for.
        length (int, optional): The length of snippets to include. Defaults to 500.
        api_key (str, optional): The API key for accessing the Serper API. Defaults to None.
        min_search (int, optional): The minimum number of search results to fetch. Defaults to 30.

    Returns:
        tuple: A tuple containing all media outlets, reliable media, and unreliable media.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": subject, "gl": "sv", "hl": "es-419", "num": min_search, "tbs": "qdr:m"})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        organic_results = results.get("organic", [])
    except requests.RequestException as e:
        logging.error(f"Error during search: {e}")
        return [], [], []

    all_media_outlets, reliable_media, unreliable_media = [], [], []

    for result in organic_results:
        snippet = result.get('snippet')
        source_url = result.get('link')
        if snippet and source_url:
            base_domain = extract_base_domain(source_url)
            all_media_outlets.append(base_domain)

            if base_domain in RELIABLE_SOURCES:
                reliability_index = RELIABLE_SOURCES[base_domain]
                reliable_media.append({
                    "snippet": snippet[:length],
                    "source": source_url,  # Include the full link
                    "reliability_index": reliability_index
                })
            else:
                unreliable_media.append({
                    "snippet": snippet[:length],
                    "source": source_url  # Include the full link
                })
        else:
            logging.warning(f"Missing data in results: {result}")

    sorted_reliable = sorted(reliable_media, key=lambda x: x['reliability_index'], reverse=True)
    return all_media_outlets, sorted_reliable, unreliable_media

# Extract social media domain for filtering purposes
def extract_social_media_domain(url):
    """
    Extract the base domain and specific path for filtering social media URLs.

    Args:
        url (str): The URL to extract the domain from.

    Returns:
        str: The domain combined with path information for social media URLs.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    if domain.startswith("www."):
        domain = domain[4:]

    # Combine domain and the first part of the path for social media filtering
    if "facebook.com" in domain or "instagram.com" in domain or "twitter.com" in domain or "x.com" in domain:
        return domain + path.split('/')[1] + '/'

    return domain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example function to log and handle errors
def handle_error(error_message):
    """
    Log an error message and handle it appropriately.

    Args:
        error_message (str): The error message to log.
    """
    logger.error(error_message)

# Example usage of logging
if __name__ == "__main__":
    try:
        # Example test of the info_extraction function
        api_key = "YOUR_SERPER_API_KEY"
        subject = "El Salvador economic growth"
        all_outlets, reliable, unreliable = info_extraction(subject, api_key=api_key)
        print("All Media Outlets:", all_outlets)
        print("Reliable Media:", reliable)
        print("Unreliable Media:", unreliable)
    except Exception as e:
        handle_error(f"An error occurred: {e}")

