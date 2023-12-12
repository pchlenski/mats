# mats-sprint
SERI MATS research sprint

Code to generate environment:
```bash
conda create -n sprint python==3.10 ipython ipykernel
conda activate sprint
pip3 install torch torchvision torchaudio
pip install transformer_lens transformers circuitsvis
pip install matplotlib plotly networkx tqdm 
pip install pytest gradio
pip install -e .
```