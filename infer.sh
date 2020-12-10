# on random latent variable
python src/infer.py --infer_random

# on encoded latent variable
python src/infer.py --infer_encoded

# run FID
# alway `pip install pytorch-fid` first
python -m pytorch_fid ./out_images_encoded/real ./out_images_encoded/gen
