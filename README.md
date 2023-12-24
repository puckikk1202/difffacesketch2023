# difffacesketch2023
This repository contains the code for the SGLDM implementation.

## Quick Start
To set up the dependencies (`python == 3.8`), run:
```bash
pip install -r requirements.txt
```
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1bciG2R_wAKlqHpuu7Y9Dzo7a0t6KAg3l/view?usp=sharing) and place the checkpoints in the specified path ./your_cktp/path/.

Test the example result by running:

```bash
python test_sketchdiff_ddim.py --ae_ckpt ./your_cktp/path/ --dm_ckpt ./your_cktp/path/ --decoder_ckpt ./your_cktp/path/
```

You can input your own selected sketch in ./sketch_example/.

To train your own SGLDM, we offer the code for preparing your dataset via Stochastic Region Abstraction. Run:

```bash
python crop_drawing.py
```

After setting your origin dataset path and output path respectively.

Finally, to start training, run:

```bash
python train_sketchdiff.py
```

