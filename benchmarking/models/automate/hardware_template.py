"""
This module contains a template class that other hardware classes should inherit from.
"""

class HardwareTemplate:
    """
    This class is a template for hardware classes. In order to create a new hardware class,
    you should inherit from this class and implement its abstract functions.
    """

    class RAMExceededError(Exception):
        """
        Exception raised when the hardware's required RAM usage exceeds its size.
        """
        pass


    class FlashExceededError(Exception):
        """
        Exception raised when the hardware's required flash usage exceeds its size.
        """
        pass


    class BoardNotFoundError(Exception):
        """
        Exception raised when the board is not found.
        """
        pass


    def __init__(self, software_platform):
        """
        Initializes the hardware class.

        Args:
            software_platform (str): The software platform that will be used with this hardware.
        """
        pass


    def get_model_dir(self):
        """
        Returns the directory where the model files can go.

        Returns:
            str: The directory where the model files can go.
        """
        raise NotImplementedError


    def build_project(self, clean=False):
        """
        Builds the project.

        Args:
            clean (bool): Whether to clean the project before building.

        Returns:
            tuple: A tuple containing text_size, data_size, and bss_size.

        Raises:
            HardwareTemplate.RAMExceededError: If the hardware's required RAM usage exceeds its size.
            Exception: If for other reasons, the project cannot be built.
        """
        raise NotImplementedError


    def upload_program(self):
        """
        Uploads the program to the hardware.

        Raises:
            Exception: If the program cannot be uploaded.
        """
        raise NotImplementedError


    @staticmethod
    def read_output(overall_timeout, silence_timeout, keyword=None, verbose=False):
        """
        Reads the output from the hardware.

        Args:
            overall_timeout (int): The overall timeout for reading.
            silence_timeout (int): The silence timeout for reading.
            keyword (str): The keyword to stop reading at.
            verbose (bool): Whether to print the output as it is read.

        Returns:
            str: The output read from the hardware.

        Raises:
            HardwareTemplate.BoardNotFoundError: If the board is not found.
            TimeoutError: If reading times out.
            Exception: If for other reasons, the output is not complete.
        """
        raise NotImplementedError
