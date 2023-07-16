import tldextract
import urlextract

extractor = urlextract.URLExtract()


def is_valid_url(url_string):
    extracted = tldextract.extract(url_string)
    return bool(extracted.domain) and bool(extracted.suffix)


def get_urls_from_text(text: str) -> list[str]:
    urls = extractor.find_urls(text)

    unique_urls = set()
    for url in urls:
        parsed_url = tldextract.extract(url).fqdn

        # Ignore IP addresses
        if parsed_url != "":
            unique_urls.add(parsed_url)

    return list(unique_urls)
