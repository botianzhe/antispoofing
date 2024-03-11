# Training Process

python -m pip install -r requirements.txt

1. download dlib file from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat and put it in faker folder.
2. modify data path in generatejson.py and run it to generate json files.
3. run train_fp16_p1.py about 50 epoches to generate best checkpoint file.
4. run train_fp16_p21.py about 50 epoches to generate best checkpoint file.
5. run train_fp16_p22.py about 50 epoches to generate best checkpoint file.
6. use the generated three checkpoint files to generate  predictions for dev and test datasets.(python eval_unet_split_tojson.py)
