
checkPoint="/home/rlcorre/IR-final-project/IR-redo/Code/checkpoints/MRS_list_med99.ckpt" # if you have pretrained weights, update the model path
embd_name="./embeddings/train_med_list_embd.npy" 
test_embd_name="./embeddings/test_med_list_embd.npy"
data_name="med"
#python3 train_embedding.py ${checkPoint} ${embd_name} ${data_name}

test_embd_name="./embeddings/test_med_list_embd.npy"
python3 test_embedding.py ${checkPoint} ${test_embd_name} ${data_name}
