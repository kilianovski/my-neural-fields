# my-neural-fields

## Quick start

Install packages
```bash
pip install -r requirements.txt
```

### Data

- Cameraman image from [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#camera)
- 41.2 Gigapixel Panorama of Shibuya in Tokyo, Japan by Trevor Dobson. Preprocessing script is located at `data/tokyo.ipynb`

## Folder structure

Final scripts are located at `notebooks/draft_01`

Main files:

 - `story_12_cnn.ipynb`: CNN experiments
 - `train_pipeline.py`: main image fitting pipeline that takes OmegaConf cfg
 - `create_figures.ipynb`: scripts for creating figures for article
 - `run_siren_sweep.py`: siren hyperparameter tunning