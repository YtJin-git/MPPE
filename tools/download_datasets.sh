CURRENT_DIR=$(pwd)

FOLDER=$1

mkdir $FOLDER

wget --show-progress -O $FOLDER/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip