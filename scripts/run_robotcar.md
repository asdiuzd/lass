## Step

```bash

cd bin

rm -rf left right rear

./test_raycasting ~/Data/robotcar-seasons/3D-models/all-merged/all.info ~/Data/robotcar-seasons/3D-models/all-merged/all.list.txt ./tmp out_centers.json out_extrinsics.json

./batch_fillhole

./test_split_robotcar_train_test

./test_blend_color ~/Data/robotcar-seasons/images/ fill_hole/ ./ jpg

```

- 拷贝 train_list.json test_list.json id2centers.json out_extrinsics.json test_timestamps.json train_timestamps.json到对应目录。

- 注意，train_list.json test_list.json对训练无效，应该用test_timestamps.json train_timestamps.json，这样才可以left right rear一组来做定位。
