# README

## Installation
 
Installer les requirements `pip install -r requirements.txt`

Il est également nécessaire d'installer [FinRL](https://github.com/AI4Finance-Foundation/FinRL#Installation)


## Scripts

- Le script principal est `main.py`. Ce script load le dataset, crée un agent, le train, le teste et calcule la matrice d'explicabilité.

Dans le script il est possible de sélectionner le type d'algorithme à utiliser (A2C, PPO)

Pour le lancer nous avons utilisé le mésocentre Moulon avec le code:

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


- Nous avons également utilisé un script `sector_explicability.py` pour générer les explicabilités sectorielles: `python sector_explicability.py`

