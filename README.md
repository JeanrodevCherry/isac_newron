# Isaac Newron _  U-NET

Internal use only, not even licensed yet

___


## Installation

```bash
git clone git@github.com:JeanrodevCherry/isac_newron.git
cd ./isac_newron/
git lfs fetch origin main
python -m venv venv
```


<details>
  <summary>Windows installation</summary>
    
```bash
source ./venv/Scripts/activate
pip install -r ./requirements.txt
```

</details>

<details>
  <summary>Linux installation</summary>

```bash
source ./venv/bin/activate
pip install -r ./requirements.txt
```
</details>

## Usage

Activate virtual environment: 

```bash
source ./venv/bin/activate # linux
source ./venv/Scripts/activate #windows
```

-filename as `-f` parameter:
example:

```bash
python ./__init__.py -f "./data/images/D03_A1_Region1_ch00.jpg
```

### Model saved as `pattern_model.pt`

Make sure the model is loaded with lfs.


## Re-train the model

You can refine and retrain the model with you own files and labeled images 

- ``/data/images``: input files
- ``/data/masks``: labeled masks for each filename

train function is under: ```./src.isac_newron.train_model``` and save it as a new .pt file