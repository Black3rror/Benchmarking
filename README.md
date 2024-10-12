# Benchmarking

Benchmarking different embedded AI tools against various models and hardware.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands
│
├── README.md            <- The top-level README for developers using this project.
│
├── pyproject.toml       <- Project configuration file
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── requirements_dev.txt <- The requirements file for reproducing the development environment
│
├── requirements_test.txt <- The requirements file for reproducing the test environment
│
├── mkdocs.yml           <- MkDocs configuration file
│
├── tests                <- Test files
│
├── docs                 <- Documentation files
│
├── benchmarking_results <- Folder for storing benchmarking results
│   │
│   └── Experiments.xlsx <- Excel file that contains the results of the experiments
│
├── benchmarking         <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── models           <- Python modules for generation, conversion, and testing of models
│   │
│   ├── Hardware         <- C/C++ projects for benchmarking hardware
│   │
│   └── main.py          <- Main script for running the benchmarking
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [DL_project_template](https://github.com/Black3rror/DL_project_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for starting a Deep Learning Project.
