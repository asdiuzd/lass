#!/usr/bin/env bash

# seq_name='OldHospital'
seq_name='KingsCollege'
# seq_name='ShopFacade'
if [ $# -eq 1 ]
  then
    seq_name=$1
fi
#dataset_base_dir=~/Data/cambridge/
dataset_base_dir=~/Dataset/CambridgeLandmarks/
output_base_dir=./cambridge_all/

cd ../bin

segmentation_dir=${output_base_dir}/${seq_name}/segmentation/
blend_color_dir=${output_base_dir}/${seq_name}/blend_color/

rm -rf ${segmentation_dir} ${blend_color}

# run raycast
./test_cambridge_landmark ../scripts/cambridge_config/${seq_name}.json ${dataset_base_dir}/${seq_name}/ ${segmentation_dir}

# run blend color
run_blend_color=true
if [ "$run_blend_color" = true ] ; then
    ./test_blend_color ${dataset_base_dir}/${seq_name}/ ${segmentation_dir} ${segmentation_dir} png
fi

# remove empty folders
find ./cambridge_all/${seq_name}/ -type d -empty -delete
