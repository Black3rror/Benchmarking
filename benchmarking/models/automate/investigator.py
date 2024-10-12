import re

import numpy as np


def find_exe_time(text):
    """
    Find the execution time from the text.

    Args:
        text (str): The text to search.

    Returns:
        tuple: A tuple of (n_tests, avg_ms, std_ms, avg_ticks, std_ticks).
    """
    ms = []
    ticks = []

    pattern = r"Execution time: ([\d.]+) ms \((\d+) ticks\)"
    matches = re.findall(pattern, text)

    for match in matches:
        ms.append(float(match[0]))
        ticks.append(int(match[1]))

    pattern = r"Execution time: (\d+) ticks"
    matches = re.findall(pattern, text)

    for match in matches:
        ticks.append(int(match))

    ms = np.array(ms)
    ticks = np.array(ticks)

    if len(ms) > 0:
        assert len(ticks) == len(ms)
    n_tests = len(ticks)

    if len(ms) > 0:
        avg_ms = np.mean(ms)
        std_ms = np.std(ms)
    else:
        avg_ms = None
        std_ms = None

    if len(ticks) > 0:
        avg_ticks = np.mean(ticks)
        std_ticks = np.std(ticks)
    else:
        avg_ticks = None
        std_ticks = None

    return n_tests, avg_ms, std_ms, avg_ticks, std_ticks


def find_prediction_mae(text):
    """
    Find the mean absolute error (MAE) from the text.

    Args:
        text (str): The text to search.

    Returns:
        tuple: A tuple of (n_tests, avg_mae, std_mae).
    """
    maes = []

    y_expected_min = None
    y_expected_max = None
    for line in text.split("\n"):
        pattern = r"\[(-?[\d.]+), (-?[\d.]+)\]"
        matches = re.findall(pattern, line)

        y_expected = []
        y_predicted = []
        for match in matches:
            y_expected.append(float(match[0]))
            y_predicted.append(float(match[1]))
            if y_expected_min is None or float(match[0]) < y_expected_min:
                y_expected_min = float(match[0])
            if y_expected_max is None or float(match[0]) > y_expected_max:
                y_expected_max = float(match[0])

        if len(y_expected) > 0:
            maes.append(np.abs(np.array(y_expected) - np.array(y_predicted)))

    maes = np.array(maes)
    if y_expected_min is not None and y_expected_max is not None and (y_expected_max - y_expected_min) > 0:
        maes = maes / (y_expected_max - y_expected_min)

    n_tests = len(maes)
    if len(maes) > 0:
        avg_mae = np.mean(maes)
        std_mae = np.std(maes)
    else:
        avg_mae = None
        std_mae = None

    return n_tests, avg_mae, std_mae
