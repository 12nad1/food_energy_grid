# food_energy_grid

## Setup Instructions

### 1. Create a Python Environment and Install Dependencies

Install SESAME-IESD and its dependencies by following the instructions at: https://github.com/A2Faisal/SESAME. Below is an example 

```bash
# create a new conda environment
conda create -n my_env
# activate the environment
conda activate my_env
# install pip
conda install pip
# install SESAME from testPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sesame-iesd==0.1.2
```

## Usage

The script can be run with various command line options:

### Basic Usage
```bash
python main.py
```
This will use the default `config.json` file in the current directory.

### Advanced Usage
```bash
# Specify a custom config file
python main.py --config path/to/your/config.json

# Run in verbose mode for detailed output
python main.py --verbose

# Generate output data (without this flag, it only visualizes existing data)
python main.py --generate_data

# Combine multiple options
python main.py --config my_config.json --verbose --generate_data
```

### Command Line Arguments

- `--config`: Path to config file (default: `config.json`)
- `--verbose`: Run in verbose mode with detailed output
- `--generate_data`: Generate the output data (without this argument, just visualizes existing data)

## Dependencies

- numpy                     2.0.1
- pandas                    2.2.2
- xarray                    2024.6.0
- cartopy                   0.23.0
- seaborn                   0.13.2
- SESAME (from https://github.com/A2Faisal/SESAME)

