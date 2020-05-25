from pathlib import Path


def check_files_existence(files: list):
    """
    Checks the existence of a list of files (all should exist)
    Arguments:
        file {Path} -- [description]

    Returns:
        [type] -- [description]
    """
    exist = all(Path(f).exists() for f in files)
    return exist
