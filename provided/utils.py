
def print_header(title: str, width: int = 50):
    """
    Prints a formatted header with a given title.

    Args:
        title (str): The title to display.
        width (int): The total width of the header (default: 50).
    """
    print("=" * width)
    centered_title = title.center(width)
    print(centered_title)
    print("=" * width)