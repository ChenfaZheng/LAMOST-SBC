# LAMOST Star-based Clibration

A toy implementation of the method, developed by Yuan et al, to calibrate the spectrum of the star observed by LAMOST.

See also: 
- Yang Sun et al 2021 Res. Astron. Astrophys. 21 092
- Hai-Bo Yuan et al 2021 Res. Astron. Astrophys. 21 074

## Usage

Simply run `main.py`. 

## Environments

Developed using python3.10, see `requirements.txt` for detail.

### A quick setup guide

1. Make sure you have [Anaconda](https://www.anaconda.com/) installed.
2. Install requirements accroding to `requirements.txt`. For example, 
    ```
    conda create --name lamost-calibration --file requirements.txt
    ```
    Check if your proxy service was disabled if something goes wrong while installing packages.
3. Make sure you're in the correct conda environment before execute the program. To activate a conda environment (e.g. `lamost-calibration`), simply execute
    ```
    conda activate lamost-calibration
    ```
    To exit an environment, execute
    ```
    conda deactivate
    ```

## License

MIT