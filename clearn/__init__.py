from pathlib import Path


def clearn_path(rel_path):
    """
    :param rel_path: POSIX path relative to the clearn root directory
    :return: string representing absolute POSIX path
    """
    return str(Path(rel_path).absolute())