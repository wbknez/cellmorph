"""
Contains classes and functions to work with emojis.
"""
from enum import IntEnum, StrEnum
from io import BytesIO
from pathlib import Path
from re import UNICODE, compile

from PIL import Image as ImageFactory
from PIL.Image import Image
from requests import get as get_content


EMOJI_URL_BASE = "https://github.com/googlefonts/noto-emoji"
EMOJI_URL_PART = "blob/main/png/{}/emoji_u{}.png?raw=true"

EMOJI_SIZES = [32, 72, 128, 512]
"""The available image sizes supported by Noto emojis."""


EMOJI_MATCHER = compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F191-\U0001F1AA"
    "\U0001F201-\U0001F21A"
    "\U0001F22F"
    "\U0001F232-\U0001F23A"
    "\U0001F250-\U0001F251"
    "\U0001F300-\U0001F5FF"  # Additional symbols and pictographs
    "\U0001F600-\U0001F64F"  # More emoticons
    "\U0001F680-\U0001F6FF"  # Transport and map symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=UNICODE
)
"""Regular expression matcher to determine if a string is an emoji."""


class CommonEmojis(StrEnum):
    """
    A collection of all common emojis used in the original paper for both
    training and demonstration purposes.
    """

    BUTTERFLY = "1f98b"
    """An orange butterfly."""

    CHRISTMAS_TREE = "1f384"
    """A Christmas tree with red and yellow ornaments."""

    EXPLOSION = "1f4a5"
    """A red, orange, and yellow explosion."""

    EYE = "1f441"
    """An eye with a brown iris."""

    FISH = "1f420"
    """A yellow fish with black stripes."""

    GECKO = "1f98e"
    """A green gecko with light facial highlights."""

    LADYBUG = "1f41e"
    """A red ladybug with black spots."""

    PRETZEL = "1f968"
    """A large brown pretzel."""

    SMILEY = "1f600"
    """A yellow smiling face."""

    SPIDERWEB = "1f578"
    """A hexagonal three-layered spiderweb with a spider in the center."""
    

class EmojiSizes(IntEnum):
    """
    A collection of generated image sizes available from the Noto font
    repository.
    """

    i32 = 32
    """A 32x32 pixel image."""

    i72 = 72
    """A 72x72 pixel image."""

    i128 = 128
    """A 128x128 pixel image."""

    i512 = 512
    """A 512x512 pixel image."""


def fetch_emoji(emoji_code: str, image_size: int = 128,
                cache_dir: Path | None = None) -> Image:
    """
    Reads an emoji as a Pillow image from the Noto Googlefonts repository.

    This function includes the ability to cache downloaded emoji images to avoid
    needing to open an HTTPS connection every run.

    Args:
        emoji_code: The hexadecimal code of the emoji to load.
        image_size: The initial size of the downloadable emoji image.
        cache_dir: The directory to cache downloaded images in; optional.
        max_size: The new desired size.

    Returns:
        An emoji as a Pillow image.
    """
    emoji_path = (cache_dir / f"{emoji_code}.png").resolve() if cache_dir \
        else None

    if emoji_path and emoji_path.exists():
        return ImageFactory.open(emoji_path)

    with get_content(make_url(emoji_code, image_size)) as req:
        img = ImageFactory.open(BytesIO(req.content))

        if emoji_path:
            img.save(emoji_path)

        return img


def is_code(emoji_code: str) -> bool:
    """
    Determines whether a standard string is a valid hexadecimal number.

    Args:
        emoji_code: The string to analyze.

    Returns:
        `True` if hexadecimal, otherwise `False`.
    """
    try:
        hex = int(emoji_code, 16)
        return True
    except ValueError:
        return False


def is_emoji(emoji: str) -> bool:
    """
    Determines whether a standard string is a valid Unicode emoji.

    Args:
        emoji: The string to check.

    Returns:
        `True` if a Unicode emoji, otherwise `False`.
    """
    return bool(EMOJI_MATCHER.fullmatch(emoji))


def make_url(emoji_code: str, image_size: int = 128) -> str:
    """
    Converts a hexadecimal code to a valid URL pointing to an image
    representation in the Noto fonts Github repository.

    Args:
        emoji_code: The hexadecimal code of the emoji to load.
        image_size: The initial size of the downloadable emoji image.

    Returns:
        A URL to a Noto fonts emoji as a standard string.
    """
    if not image_size in EMOJI_SIZES:
        raise ValueError(f"Unknown image size: {image_size}.")

    return f"{EMOJI_URL_BASE}/{EMOJI_URL_PART.format(image_size,
                                                     emoji_code.lower())}"

def to_code(emoji: str) -> str:
    """
    Converts an emoji - a single character with a non-alphabetic representation
    - to a hexadecimal code value.

    Args:
        emoji: The emoji to convert.

    Returns:
        A hexadecimal code representation as a string.

    Raises:
        ValueError: If `emoji` has a length greater than one.
    """
    if len(emoji) > 1:
        # Likely already converted to hex, otherwise just invalid input.
        raise ValueError(f"String is likely not an emoji: {emoji}.")

    return hex(ord(emoji))[2:]


def to_emoji(emoji_code: str) -> str:
    """
    Converts a hexadecimal code value to a single Unicode emoji character.

    """
    return chr(int(emoji_code, 16))
