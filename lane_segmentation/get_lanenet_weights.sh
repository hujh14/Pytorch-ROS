
mkdir lanenet-lane-detection/model
mkdir lanenet-lane-detection/model/tusimple_lanenet
cd lanenet-lane-detection/model/tusimple_lanenet

wget "https://www.dropbox.com/sh/tnsf0lw6psszvy4/AADK1Dqlbmzbz2cCTJiSzAWha/checkpoint"
wget "https://www.dropbox.com/sh/tnsf0lw6psszvy4/AADKYYa2ou8VhqDcSNCN69pVa/lanenet_model.pb"
wget "https://www.dropbox.com/sh/tnsf0lw6psszvy4/AACBMPUKhUl1B3kTeyhBIS4Ja/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000.data-00000-of-00001"
wget "https://www.dropbox.com/sh/tnsf0lw6psszvy4/AADY04f7wHs_lwuCobMKVM0ta/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000.index"
wget "https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAD9EMV7sbdsHcFTfEFsqWIGa/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000.meta"

cd ../../..