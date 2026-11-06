"""Text representation of a window, shared by Scenario 2, Scenario 2b and LUK: the set of distinct templates of the window, with no
counts, ordered by Drain template id, joined by the [SEP] token. The order of appearance and the number of lines do not change it."""

SEPARATOR = " [SEP] "
MAX_TOKENS = 512


class WindowTooLong(Exception):
    pass


def window_text(template_ids, templates):
    """template_ids: ids of the lines of a window; templates: id -> text of the view."""
    return SEPARATOR.join(templates[template_id] for template_id in sorted(set(template_ids)))


def token_counts(texts, tokenizer):
    """Tokens of each text with [CLS] and the final [SEP], without truncation."""
    return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=True, truncation=False, padding=False)["input_ids"]]


def check_limit(counts, names, limit=MAX_TOKENS):
    """No truncation: a window above the limit stops everything."""
    over = [(name, count) for name, count in zip(names, counts) if count > limit]
    if over:
        raise WindowTooLong(f"{len(over)} windows above {limit} tokens, longest {max(c for _, c in over)}, e.g. {over[0][0]}")
