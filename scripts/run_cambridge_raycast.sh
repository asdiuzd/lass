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

cd ../bin

rm -rf cambridge_raycast
rm -rf cambridge_all/${seq_name}

for ((i=1;i<=25;i++)); do
    mkdir -p cambridge_raycast/seq$i
done

mkdir -p cambridge_raycast/img cambridge_raycast/img_east cambridge_raycast/img_west cambridge_raycast/img_south cambridge_raycast/img_north


# run raycast
./test_cambridge_landmark ../scripts/cambridge_config/${seq_name}.json ${dataset_base_dir}/${seq_name}/

# copy parameters
cp train_list.json cambridge_raycast/
cp test_list.json cambridge_raycast/
cp test_list.json cambridge_raycast/
cp id2centers.json cambridge_raycast/
cp out_extrinsics.json cambridge_raycast/


# move to standalone folders
mkdir -p cambridge_all/${seq_name}
mv  cambridge_raycast cambridge_all/${seq_name}/raycast/

# run blend color
run_blend_color=true
if [ "$run_blend_color" = true ] ; then
    for ((i=1;i<=25;i++)); do
        mkdir -p cambridge_all/${seq_name}/blend_color/seq$i
    done
    cd cambridge_all/${seq_name}/blend_color
    mkdir -p img img_east img_west img_north img_south
    cd ../../..

    ./test_blend_color ${dataset_base_dir}/${seq_name}/ cambridge_all/${seq_name}/raycast cambridge_all/${seq_name}/raycast png
fi

# remove empty folders
find ./cambridge_all/${seq_name}/ -type d -empty -delete
