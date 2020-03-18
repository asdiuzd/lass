#!/usr/bin/env bash

cd ../bin

dataset_base_dir=~/Data-2T/aachen/
output_base_dir=./aachen/

segmentation_dir=${output_base_dir}/segmentation/
blend_color_dir=${output_base_dir}/blend_color/

rm -rf ${output_base_dir}
mkdir -p ${output_base_dir}

./test_aachen_daynight ../scripts/aachen_daynight.json ${dataset_base_dir} ${segmentation_dir}

run_blend_color=true
if [ "$run_blend_color" = true ] ; then
    ./test_blend_color ${dataset_base_dir} ${segmentation_dir} ${segmentation_dir} png
fi

# remove empty folders
find ./cambridge_all/ -type d -empty -delete