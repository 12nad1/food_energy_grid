# food_energy_grid

This is the code repository for the paper Energetic Closure of the Spatially Resolved Global Food System, which is currently under review at Nature Food, with preprint available here: https://arxiv.org/abs/2412.10421.

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
# install cvxpy
pip install cvxpy==1.7.1
```

### 2. Download data

All data used in this resesarch was open source and openly available. The input data and the output data are available at the Zenodo directory at URL https://zenodo.org/records/16708221 with DOI 10.5281/zenodo.16708221.

If users want to reproduce the work from the paper, they can download the input data, contained in data.zip. If users simply want the output data from the paper, it is in the directory output.zip. It is recomended that these files be unzipped within their own directories data and output, respectively, for easy file management and references from the config.json file.

The original citations for the data in data.zip can be found in the paper. Please cite the original source if using this data directly. Note that the file fao_country_to_region.json is already in the data directory in this git repository, however it is not available in the published zenodo directory. 

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

Note that running the full data generation takes several hours

### Command Line Arguments

- `--config`: Path to config file (default: `config.json`)
- `--verbose`: Run in verbose mode with detailed output
- `--generate_data`: Generate the output data (without this argument, just visualizes existing data)

## Dependencies

- cvxpy                     1.7.1
- sesame_iesd 				0.1.2	(from https://github.com/A2Faisal/SESAME)

