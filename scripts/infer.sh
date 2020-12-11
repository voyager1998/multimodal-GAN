if [ -d "data/edges2shoes" ] 
    then
    echo "Dataset exists. No need to download dataset."
else
    echo "Downloading dataset. This may take a while..."
    cd data
    sh download_dataset.sh edges2shoes
    cd ..
fi

# on random latent variable
echo "------------- Generating inference results on test dataset..."
python src/infer.py --infer_random --show_loss_curves

# on encoded latent variable
echo "------------- Generating FID images on test dataset..."
python src/infer.py --infer_encoded

# run FID
# alway `pip install pytorch-fid` first
echo "------------- Computing FID score..."
pip install pytorch-fid
python -m pytorch_fid ./out_images_fid/real ./out_images_fid/gen > log/fid.txt
echo "------------- FID score finished"

# on LPIPS image diversity measurement
echo "------------- Generating random samples across all test dataset for LPIPS computing..."
python src/infer.py --compute_lpips
echo "------------- LPIPS images generated"

sh scripts/compute_lpips.sh


