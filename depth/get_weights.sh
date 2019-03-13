
mkdir MegaDepth/checkpoints
mkdir MegaDepth/checkpoints/test_local
cd MegaDepth/checkpoints/test_local

wget http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_vanila_net_G.pth
wget http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth
wget http://www.cs.cornell.edu/projects/megadepth/dataset/models/test_model_1_4.zip

unzip test_model_1_4.zip

cd ../../..