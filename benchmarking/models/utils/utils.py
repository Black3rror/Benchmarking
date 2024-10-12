import os


def get_abs_path(relative_path):
    """
    Get the absolute path of the given relative path to the root directory of the project.

    Args:
        relative_path (str): The relative path to the root directory of the project.

    Returns:
        str: The absolute path of the given relative path to the root directory of the project.
    """
    return os.path.abspath(relative_path).replace('\\', '/')


def find_target_files(dir):
    """
    Find all target files in the given directory and its subdirectories.
    The files and directories starting with '.' are ignored.

    Args:
        dir (str): The directory to search for target files.

    Returns:
        list: A list of relative paths to the target files.
    """

    target_files = []

    for dirpath, dirnames, filenames in os.walk(dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        for filename in filenames:
            if filename.startswith('.'):
                continue

            if filename.endswith('.yaml') or filename.endswith('.yml'):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, start=dir)
                relative_path = relative_path.replace('\\', '/')
                target_files.append(relative_path)

    return target_files
