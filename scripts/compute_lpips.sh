echo "------------- Computing LPIPS score..."
# calculate LPIPS
pip install lpips
python src/lpips_all_dir.py -d out_images_lpips/ -o log/dists_pair.txt --use_gpu