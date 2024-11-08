import os
import re
import subprocess
import time

import serial
import serial.tools.list_ports
from omegaconf import OmegaConf

from edgemark.models.automate.hardware_template import HardwareTemplate
from edgemark.models.utils.utils import get_abs_path


config_file_path = "edgemark/models/automate/hardware_types/NUCLEO-L4R5ZI/configs/hardware_config.yaml"
config_file_path = get_abs_path(config_file_path)


class Hardware(HardwareTemplate):
    def __init__(self, software_platform):
        super().__init__(software_platform)

        self.cfg = OmegaConf.load(config_file_path)
        if os.path.exists(self.cfg.user_config):
            self.cfg = OmegaConf.merge(self.cfg, OmegaConf.load(self.cfg.user_config))
        self.cfg.project_name = "NUCLEO-L4R5ZI_{}".format(software_platform)


    def get_model_dir(self):
        return os.path.join(self.cfg.project_dir, "Core/Src/model")


    def build_project(self, clean=False):
        task = "-build" if not clean else "-cleanBuild"
        command = [
            self.cfg.stm32cubeide_path,
            "-nosplash", "--launcher.suppressErrors",
            "-application", "org.eclipse.cdt.managedbuilder.core.headlessbuild",
            "-data", self.cfg.workspace_dir,
            task, self.cfg.project_name
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        text_size, data_size, bss_size = Hardware._find_memory_info(result.stdout)
        build_success = True if text_size is not None else False
        if build_success:
            return text_size, data_size, bss_size
        else:
            report = "Build failed.\nBuild status: {}\n\n\nBuild stdout: {}\n\n\nBuild stderr: {}\n".format(result.returncode, result.stdout, result.stderr)
            ram_error_txt_1 = "will not fit in region `RAM'"
            ram_error_text_2 = "is not within region `RAM'"
            flash_error_txt_1 = "will not fit in region `FLASH'"
            flash_error_txt_2 = "is not within region `FLASH'"
            if ram_error_txt_1 in result.stdout or ram_error_text_2 in result.stdout:
                raise Hardware.RAMExceededError(report)
            elif flash_error_txt_1 in result.stdout or flash_error_txt_2 in result.stdout:
                raise Hardware.FlashExceededError(report)
            else:
                raise Exception(report)


    def upload_program(self):
        elf_path = os.path.join(self.cfg.project_dir, "Debug", os.path.basename(self.cfg.project_dir) + ".elf")
        elf_path = get_abs_path(elf_path)

        command = [self.cfg.stm32_programmer_path, "-c", "port=SWD", "-e", "all"]
        result = subprocess.run(command, cwd=self.cfg.project_dir, capture_output=True, text=True)

        clean_success = True if result.returncode == 0 else False

        if not clean_success:
            report = "Erasing failed.\nErasing status: {}\n\n\nErasing stdout: {}\n\n\nErasing stderr: {}\n".format(result.returncode, result.stdout, result.stderr)
            raise Exception(report)

        command = [self.cfg.stm32_programmer_path, "-c", "port=SWD", "-w", elf_path, "-s"]
        result = subprocess.run(command, cwd=self.cfg.project_dir, capture_output=True, text=True)

        programming_success = True if result.returncode == 0 else False

        if not programming_success:
            report = "Programming failed.\nProgramming status: {}\n\n\nProgramming stdout: {}\n\n\nProgramming stderr: {}\n".format(result.returncode, result.stdout, result.stderr)
            raise Exception(report)


    @staticmethod
    def read_output(overall_timeout, silence_timeout, keyword=None, verbose=False):
        port = Hardware._find_port()
        if port is None:
            raise Hardware.BoardNotFoundError("Board not found.")

        ser = serial.Serial(port, baudrate=115200, timeout=1)

        text = ""
        try:
            overall_tic = time.time()
            silence_tic = time.time()
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()   # with the timeout that is set previously
                if line:
                    text += line + "\n"
                    silence_tic = time.time()
                    if verbose:
                        print(line)

                if keyword and keyword in line:
                    break

                if overall_timeout and time.time() - overall_tic > overall_timeout:
                    raise TimeoutError("Overall timeout reached.\nRead output so far:\n{}".format(text))

                if silence_timeout and time.time() - silence_tic > silence_timeout:
                    raise TimeoutError("Silence timeout reached.\nRead output so far:\n{}".format(text))

        finally:
            ser.close()

        return text


    @staticmethod
    def _find_memory_info(text):
        """
        Find the memory information from the text.

        Args:
            text (str): The text to search for the memory information.

        Returns:
            tuple: A tuple containing the text, data, and bss sizes. If the sizes are not found, they are set to None.
        """
        pattern = re.compile(r"text\s+data\s+bss\s+dec.*\n\s*(\d+)\s+(\d+)\s+(\d+)")
        match = pattern.search(text)
        if match:
            text_size = int(match.group(1))
            data_size = int(match.group(2))
            bss_size = int(match.group(3))
        else:
            text_size = None
            data_size = None
            bss_size = None

        return text_size, data_size, bss_size


    @staticmethod
    def _find_port(verbose=False):
        """
        Find the port connected to the board.

        Args:
            verbose (bool): Whether to print the description of the available ports.

        Returns:
            str: The port connected to the board. None if the board is not found.
        """
        available_ports = list(serial.tools.list_ports.comports())

        for port in available_ports:
            if verbose:
                print(port.description)
            if "STLink" in port.description:
                return port.device

        return None
