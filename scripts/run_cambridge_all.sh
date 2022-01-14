#!/usr/bin/env bash

# ./run_cambridge_raycast.sh "ShopFacade"
./run_cambridge_raycast.sh "OldHospital"
./run_cambridge_raycast.sh "KingsCollege"
./run_cambridge_raycast.sh "GreatCourt"
./run_cambridge_raycast.sh "StMarysChurch"

cp -r /home/zhouhan/data/code/computer-vision/visual-localization/lass_backup/cambridge_all/ShopFacade/segmentation/* ~/data/cambridge_release/ShopFacade/
cp -r /home/zhouhan/data/code/computer-vision/visual-localization/lass_backup/cambridge_all/OldHospital/segmentation/* ~/data/cambridge_release/OldHospital/
cp -r /home/zhouhan/data/code/computer-vision/visual-localization/lass_backup/cambridge_all/KingsCollege/segmentation/* ~/data/cambridge_release/KingsCollege/
cp -r /home/zhouhan/data/code/computer-vision/visual-localization/lass_backup/cambridge_all/GreatCourt/segmentation/* ~/data/cambridge_release/GreatCourt/
cp -r /home/zhouhan/data/code/computer-vision/visual-localization/lass_backup/cambridge_all/StMarysChurch/segmentation/* ~/data/cambridge_release/StMarysChurch/
