if [ -d "data/edges2shoes" ] 
    then
    echo "Dataset exists. No need to download dataset."
else
    echo "Downloading dataset. This may take a while..."
    cd data
    sh download_dataset.sh edges2shoes
    cd ..
fi

python src/train.py