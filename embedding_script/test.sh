checkPoint="/home/rlcorre/IR-final-project/IR-redo/checkpoints/MRS_tiny_list_128_19.ckpt" # if you have pretrained weights, update the model path
data_name="tiny"
embd_name="./embeddings/train_${data_name}_list_embd_128.npy" 
test_embd_name="./embeddings/test_${data_name}_list_embd_128.npy"
data_path="/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/val/"
#data_path="/labs/sharmalab/cbir/dataset2/"
python3 ./src/test_embedding.py ${checkPoint} ${test_embd_name} ${data_name} "128" ${data_path}
