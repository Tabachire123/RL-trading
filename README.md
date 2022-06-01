# README

## Installation
 
Install the requirements `pip install -r requirements.txt`

It is also necessary to install [FinRL](https://github.com/AI4Finance-Foundation/FinRL#Installation)


## Scripts

- The main script is `main.py`. This script loads the dataset, creates an agent, trains it, tests it and calculates the explainability matrix.

In the script it is possible to select the type of algorithm to use (A2C, PPO)

To run it we used the Moulon mesocenter with the code:

```shell
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=rl
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2021.05/gcc-9.2.0

./main.py
```


- We also used a `sector_explicability.py` script to generate sectoral explainabilities: `python sector_explicability.py`

## Documentation

We used Sphinx to generate html documentation from the code. This documentation can be found in the `html` folder. To open it you need to download the folder and open the `index.html` file in the root.

Here are some screenshots of the result:

![example 1](/html/ex1.png)

![example 2](/html/ex2.png)
