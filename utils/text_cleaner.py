import re


def clean_message(message):
    """
    Remove numbering, bullets and extra spaces
    """

    message = re.sub(r'^\d+[\).\-\s]*', '', message)

    message = re.sub(r'^[•\-\*]\s*', '', message)

    return message.strip()