**Cellpose_ResNet.ipynb** : Generate cell segmentation masks and run it with ResNet

**CPJUMP1_simple_without_batchfile_406.cppipe** : A pipeline for generating cellular features from CellProfiler

**read_feat.ipynb** : Compare the structure of our feature profiles with the ones downloaded from the paper's github

**merge_feat.ipynb** : Average the features by sites and merge with metadata

  * metadata : *metadata_BR00116991_filtered.csv*
  
  * averaged data : *averaged_features.csv*
  
  * merged data : *merged_metadata_and_features.csv*
  
**train.py**: Training the EfficienctNet without using feature data as ablation

**train_w_feat.py** : Training the EfficienctNet with using feature data
