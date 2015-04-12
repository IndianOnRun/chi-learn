import os


def clearn_path(rel_path):
    """
    :param rel_path: POSIX path relative to the clearn root directory
    :return: string representing absolute POSIX path
    """
    directory = os.path.dirname(__file__)
    return os.path.join(directory, rel_path)