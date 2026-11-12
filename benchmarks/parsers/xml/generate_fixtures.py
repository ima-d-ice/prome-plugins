"""Generate deterministic synthetic XML fixtures."""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def make_catalog_xml() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<catalog name="TechBooks">
    <book id="bk101">
        <author>Gambardella, Matthew</author>
        <title>XML Developer's Guide</title>
        <genre>Computer</genre>
        <price currency="USD">44.95</price>
        <publish_date>2000-10-01</publish_date>
        <description>An in-depth look at creating applications with XML.</description>
    </book>
    <book id="bk102">
        <author>Ralls, Kim</author>
        <title>Midnight Rain</title>
        <genre>Fantasy</genre>
        <price currency="USD">5.95</price>
        <publish_date>2000-12-16</publish_date>
        <description>A former architect battles corporate zombies.</description>
    </book>
</catalog>
"""
    (FIXTURES / "catalog.xml").write_text(xml.strip())


def make_namespaces_xml() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<root xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:meta="http://example.com/meta">
    <dc:title>Document With Namespaces</dc:title>
    <dc:creator>Alice Developer</dc:creator>
    <meta:info version="2.0">
        <meta:tag>production</meta:tag>
    </meta:info>
</root>
"""
    (FIXTURES / "namespaces.xml").write_text(xml.strip())


def make_cdata_xml() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<article>
    <headline>Code Snippets Inside CDATA</headline>
    <content><![CDATA[function test() { return x < y && a > b; }]]></content>
</article>
"""
    (FIXTURES / "cdata.xml").write_text(xml.strip())


def make_unicode_xml() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<international>
    <greeting lang="fr">Bonjour le monde, café!</greeting>
    <greeting lang="ja">こんにちは世界</greeting>
    <greeting lang="el">Καλημέρα κόσμε</greeting>
    <greeting lang="ru">Привет мир</greeting>
</international>
"""
    (FIXTURES / "unicode.xml").write_text(xml.strip())


def make_empty_xml() -> None:
    (FIXTURES / "empty.xml").write_text("")


def make_malformed_xml() -> None:
    (FIXTURES / "malformed.xml").write_text("<root><unclosed>text</root>")


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_catalog_xml()
    make_namespaces_xml()
    make_cdata_xml()
    make_unicode_xml()
    make_empty_xml()
    make_malformed_xml()
    print(f"Generated XML fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
