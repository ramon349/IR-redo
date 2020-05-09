# IR final project
This section deals with the use of the list  loss function to  train a  neural net to generate an embedding of  images. The  embedding of images is thus used to idenitfy images that are similar  for image search purposes 

# Step 1: Pre train on tiny-imagenet 
run the scripts in train_tiny_scripts to generate an initial model that learns "shape" features and such to distinguish between natural images 

# Step 2: Train on medical images 
run train_med_scripts to run training on the medical image dataset. Should provide a prevState variable to the bash scripts pointing to the last checkpoitn of step 1

# Step 3. Generate embeddigns 
Run bash script that loads model from step2 and generates embeddings of the data in the embeddigns folder 

# Step 4: evaluation 
 TODO 
		1. MAP across categories 
		2. gradCAM visualization of activation layers 
		3. TSNE plots of embedding space? 
