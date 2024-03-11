# Training Process

1. modify data path in generatejson.py and run it to generate json files.
2. run train_fp16_p1.py about 50 epoches to generate best checkpoint file.
3. run train_fp16_p21.py about 50 epoches to generate best checkpoint file.
4. run train_fp16_p22.py about 50 epoches to generate best checkpoint file.
5. use the generated three checkpoint files to generate  predictions for dev and test datasets.(python eval_unet_split_tojson.py)
6. modify dataset.py to add the generated predictions to training dataset.(uncomment the line in "if self.dataselect == 'train':")
7. re-train the three models respectively.
8. repeat step 6 and 7 several times(about 10 times.)
9. use the final three checkpoint files to generate prediction.
