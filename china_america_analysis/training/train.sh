export WANDB_API_KEY='0e77f7c02e33b86269ca2123964b9fefcf9c1a7a'

# Loop over the years from 1990 to 2019
for YEAR in {1990..2019}
do
    INPUT_FILE='/data/align-anything/boyuan/core_workspace/raw_datasets/coca/text/text/text_news_'$YEAR'.txt'
    
    echo "Training Word2Vec model for year $YEAR with input file $INPUT_FILE"

    python dev_train_word2vec.py \
        --input_file $INPUT_FILE \
        --year $YEAR
    
    # Optional: Include a wait time between each year if needed
    # sleep 1
done
