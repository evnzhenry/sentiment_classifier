"""Text preprocessing utilities: normalize, remove URLs, mentions, hashtags, emojis, slang mapping."""
import re
import unicodedata
try:
    from emoji import demojize as _demojize
except Exception:
    # emoji package not available; fallback to identity
    def _demojize(x):
        return x

SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "plz": "please",
    "pls": "please",
    "lol": "laughing",
    "omg": "oh my god",
    # add more as needed
}

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
REPEAT_RE = re.compile(r'(\w)\1{2,}')

def normalize_unicode(text):
    if not isinstance(text, str):
        return ''
    return unicodedata.normalize('NFKC', text)

def remove_urls(text):
    return URL_RE.sub('', text)

def remove_mentions(text):
    return MENTION_RE.sub('@USER', text)

def replace_hashtags(text):
    # keep the tag word but remove hash sign
    return HASHTAG_RE.sub(lambda m: m.group(1), text)

def demojize_text(text):
    return _demojize(text)

def collapse_repeated_chars(text):
    # replace repeated characters (so cooool -> cool)
    return REPEAT_RE.sub(lambda m: m.group(1)*2, text)

def slang_normalize(text, slang_map=SLANG_MAP):
    tokens = text.split()
    out = [slang_map.get(t.lower(), t) for t in tokens]
    return ' '.join(out)

def simple_clean(text):
    t = normalize_unicode(text)
    t = remove_urls(t)
    t = remove_mentions(t)
    t = replace_hashtags(t)
    t = demojize_text(t)
    t = collapse_repeated_chars(t)
    t = slang_normalize(t)
    # normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

if __name__ == '__main__':
    s = "OMG!!! this is soooo cooool 😂😂 https://t.co/abc @john #excited"
    print('Before:', s)
    print('After :', simple_clean(s))
