"""Turning a user query into a safe FTS5 ``MATCH`` expression.

FTS5's ``MATCH`` syntax is not SQL: a query like ``AND`` or ``"`` or ``*``
is a *syntax*, not a word, and passing it through would either fail or
change the meaning of the search.  The query is therefore turned into a
sequence of quoted phrases.

**Tokenization.**  The index uses the ``unicode61`` tokenizer with
``tokenchars '_'``, so the underscore is *not* a separator:

===========================  =========================
text                         tokens
===========================  =========================
``get_user_id``              ``get_user_id``
``snake_case``               ``snake_case``
``HTTPStatusCode``           ``httpstatuscode``
``/api/v1/users``            ``api`` ``v1`` ``users``
``error E_TIMEOUT``          ``error`` ``e_timeout``
``value=42``                 ``value`` ``42``
===========================  =========================

A query is tokenized by FTS5 with the *same* rules as the index, so a query
and a document agree on what a token is.  A query term is quoted before it
is sent, so a term that contains FTS5 syntax is searched as the literal
word it is.

The terms of a query are combined with ``AND``: a document must contain
every term of the query to match.  That is the behaviour of the FTS5
``unicode61`` tokenizer's implicit conjunction and it is what makes an
exact-identifier query precise.
"""

from __future__ import annotations

import re

#: Characters that FTS5 treats as syntax inside a bare word.
#:
#: They are removed from a term before it is quoted, because a quoted phrase
#: may not contain them as syntax anyway.
_SPECIAL = re.compile(r'[":^*(){}[\]-]')

#: The characters that separate the words of a query.
_WORDS = re.compile(r"[^\w]+", re.UNICODE)


def tokenize_query(text: str) -> list[str]:
    """Return the terms of ``text``, in the order they appear.

    The terms are *not* lower-cased: the FTS5 ``unicode61`` tokenizer does
    that itself, and doing it twice would change the behaviour of a
    case-sensitive query.
    """
    if not text:
        return []
    terms: list[str] = []
    for word in _WORDS.split(text):
        cleaned = _SPECIAL.sub("", word).strip()
        if cleaned:
            terms.append(cleaned)
    return terms


def quote_term(term: str) -> str:
    """Return ``term`` as a quoted FTS5 phrase.

    The double quotes inside the term are doubled, which is how FTS5 escapes
    them, so a term can never break out of the phrase.
    """
    return '"' + term.replace('"', '""') + '"'


def build_match_query(text: str) -> str:
    """Return the FTS5 ``MATCH`` expression for ``text``.

    Returns an empty string when the query holds no searchable term; the
    caller must then return no result rather than match everything.
    """
    terms = tokenize_query(text)
    if not terms:
        return ""
    return " AND ".join(quote_term(term) for term in terms)


def build_prefix_match_query(text: str, *, prefix: bool = True) -> str:
    """Return a ``MATCH`` expression that also matches the prefixes.

    A trailing ``*`` turns the last term into a prefix term, which finds a
    document that starts with the term even when the term is longer there
    (``authen`` matches ``authentication``).
    """
    terms = tokenize_query(text)
    if not terms:
        return ""
    quoted = [quote_term(term) for term in terms]
    if prefix:
        quoted[-1] = quoted[-1] + "*"
    return " AND ".join(quoted)
