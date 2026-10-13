"""Domain-independent grammar helpers for question-focused retrieval."""

import re


RETRIEVAL_POLICY = "parent-question-focus-component-reservation-v2"
COVERAGE_POLICY = "semantic-with-discriminative-anchors-v2"

QUESTION_STARTERS = (
    "can", "could", "do", "does", "did", "is", "are", "what", "which",
    "how", "where", "when", "who", "why", "would", "will", "should",
)
_QUESTION_START = re.compile(
    r"^(?:" + "|".join(QUESTION_STARTERS) + r")\b", re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(r"[,;:\r\n]+|(?<=[.!?])\s+")

# Grammatical function words delimit phrases without interpreting their topic.
# Content words, including ordinal adjectives and repetition adverbs, remain.
_FUNCTION_WORDS = frozenset("""
    a an the this that these those any each every some no
    i me my mine myself you your yours yourself yourselves
    he him his himself she her hers herself it its itself
    we us our ours ourselves they them their theirs themselves
    what which how where when who whom whose why
    am is are was were be been being do does did have has had
    can could may might must shall should will would cannot
    about above across after against along among around as at before behind
    below beneath beside between beyond by despite during except for from
    in inside into near of off on onto out outside over past per since
    through throughout till to toward towards under underneath until up
    upon via with within without
    and but if nor or so than though unless whether while yet not
""".split())
_WORD_PATTERN = r"\w+(?:['’\-]\w+)*"
_TOKENS = re.compile(_WORD_PATTERN + r"|[^\w\s]")
_WORD = re.compile(_WORD_PATTERN)


def question_focus(query: str) -> str:
    """Select the first explicit question clause, preserving the whole if absent."""
    whole = query.strip()
    # A question may end with a comma-separated qualifier; retain that context.
    if _QUESTION_START.match(whole):
        return whole
    for candidate in _CLAUSE_BOUNDARY.split(whole):
        clause = candidate.strip(" \t\r\n\"'“”‘’")
        if _QUESTION_START.match(clause):
            return clause if clause != whole else whole
    return whole


def _is_function_word(word: str) -> bool:
    normalized = word.casefold().replace("’", "'")
    if normalized in _FUNCTION_WORDS:
        return True
    if normalized in {"can't", "won't", "shan't"}:
        return True
    if normalized.endswith("n't") and normalized[:-3] in _FUNCTION_WORDS:
        return True
    if "'" in normalized:
        base, suffix = normalized.rsplit("'", 1)
        if base in _FUNCTION_WORDS and suffix in {"m", "re", "ve", "ll", "d", "s"}:
            return True
    return False


def question_components(query: str) -> list[str]:
    """Return unique contiguous content phrases from the focused question."""
    components = []
    words = []

    def finish_phrase() -> None:
        if words:
            phrase = " ".join(words)
            if phrase not in components:
                components.append(phrase)
            words.clear()

    for token in _TOKENS.findall(question_focus(query)):
        if not _WORD.fullmatch(token) or _is_function_word(token):
            finish_phrase()
        else:
            words.append(token.casefold().replace("’", "'"))
    finish_phrase()
    return components


def component_anchor(component: str, text: str, documents: dict[str, str]) -> bool:
    """Anchor on the least common observed content word in a phrase.

    A shared generic word must not substitute for the phrase's distinguishing
    word. Unseen paraphrase words do not veto its observed textual anchor.
    """
    patterns = {word: re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                for word in component.split()}
    frequencies = {word: sum(bool(pattern.search(body)) for body in documents.values())
                   for word, pattern in patterns.items()}
    observed = {word: count for word, count in frequencies.items() if count}
    if not observed:
        return False
    rarest = min(observed.values())
    return any(patterns[word].search(text) for word, count in observed.items()
               if count == rarest)
