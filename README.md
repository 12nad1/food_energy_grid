# food_energy_grid

This is the code repository for the paper Energetic Closure of the Spatially Resolved Global Food System, which is currently under review, with preprint available here: https://arxiv.org/abs/2412.10421.

## Setup Instructions

### 1. Create a Python Environment and Install Dependencies
(may take a few minutes)

Install SESAME-IESD and its dependencies by following the instructions at: https://github.com/A2Faisal/SESAME (typically takes a few minutes to install). Below is an example 

```bash
# create a new conda environment
conda create -name my_env
# activate the environment
conda activate my_env
# install pip
conda install pip
## install SESAME from testPyPI
#pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sesame-iesd==0.1.2
# install SESAME from pip
pip install sesame-iesd==1.0.1
# install cvxpy and h5py
pip install cvxpy==1.7.1
pip install h5py==3.16.0
```

### 2. Download data
(may take a few minutes)

All data used in this resesarch was openly available. The input data and the output data are available at the Zenodo directory at URL https://zenodo.org/records/16708221 with DOI 10.5281/zenodo.16708221.

If users want to reproduce the work from the paper, they can download the input data, contained in data.zip, and add it to the data directory. Users can also use the output data from the paper, found in the directory contained in output.zip. It is recomended that these files be unzipped within their own directories data and output, respectively, for easy file management and references from the config.json file.

The original citations for the data in data.zip can be found in the paper. Please cite the original source if using this data directly. Note that the file fao_country_to_region.json is already in the data directory in this git repository, however it is not available in the published zenodo directory. 

### 3. Modify for use case

Some users will be interested in simply downloading the output of the analysis, which can be found at the link above. Others will be interested in running the code to generate new output based on a modified set of parameters. The input paths can be modified in the config.json file, and as long as the metadata match, new csv files and netcdf files can be plugged into the pipeline. For example, users might prefer to download an FAO Food Balance Sheet from a different year and substititute its path in the config file. Or perhaps other users would prefer to use a different surrogate netcdf file which was not available or not used in the original study. 

## Usage

The script can be run with various command line options:

### Basic Usage
```bash
python main.py
```
This will use the default `config.json` file in the current directory. This can be used as a quick demo to generate the data from the output files, and will only take a few minutes.

### Advanced Usage
(note that the --generate_data flag the code may take several hours)

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

Note that running the full data generation can take several hours, however the visualization and analysis can take a few minutes and the generated data can be downloaded.

### Command Line Arguments

- `--config`: Path to config file (default: `config.json`)
- `--verbose`: Run in verbose mode with detailed output
- `--generate_data`: Generate the output data (without this argument, just visualizes existing data)

## Dependencies

- cvxpy                     1.7.1
- sesame_iesd 				0.1.2	(from https://github.com/A2Faisal/SESAME)

