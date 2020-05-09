
checkPoint="/home/rlcorre/IR-final-project/IR-redo/Code/checkpoints/MRS5.ckpt" # if you have pretrained weights, update the model path
embd_name="train_triplet_embd.npy"
python3 train_embedding.py ${checkPoint} ${embd_name}
test_embd_name="test_triplet_embd.npy"
python3 test_embedding.py ${checkPoint} ${test_embd_name}
