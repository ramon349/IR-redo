checkPoint="/home/rlcorre/IR-final-project/IR-redo/checkpoints/MRS_med_list_128_99.ckpt" # if you have pretrained weights, update the model path
data_name="med"
embd_name="./embeddings/train_${data_name}_list_embd_128.npy" 
test_embd_name="./embeddings/test_${data_name}_list_embd_128.npy"
#data_path="/home/rlcorre/IR-final-project/deep-raeaking-master/tiny-imagenet-200/"
data_path="/labs/sharmalab/cbir/dataset2"
echo "Generating train embedding" 
python3 ./src/train_embedding.py ${checkPoint} ${embd_name} ${data_name} "128" ${data_path}
data_path="/labs/sharmalab/cbir/dataset2/val"
echo "Generating train embedding" 
python3 ./src/test_embedding.py ${checkPoint} ${test_embd_name} ${data_name} "128" ${data_path}
