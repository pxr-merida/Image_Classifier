# Image_Classifier
Find classes of flowers from images 

Image Classifier code is provided with Image Classifier Project_Part1.ipynb. 

You can also use train.py and predict.py as follows:

python train.py flowers --gpu --epoch 5 --arch densenet121 --learning_rate 0.001 --hidden_units 500 

python predict.py flowers/test/43/image_02365.jpg checkpnt.pth --gpu

You can view output with Fig1.png and Prob_Classname.csv file.
