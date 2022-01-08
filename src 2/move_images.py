import os
import csv

with open('../data/trainLabels.csv') as file:
    reader = csv.reader(file)
    
    for row in reader:
        image = row[0] + '.jpeg'
        status= row[1]
        
        if status is '0':
            try :
                os.rename('../data/train-resized-224/'+ image, '../data/train-resized-224/retinopathy/'+ image)
                print('moved to retinopathy')
            except FileNotFoundError:
                print('did not read file')
                
        if status in ['1','2','3','4']:
            try:
                os.rename('../retinopathy/train/'+ image, '../retinopathy/train/noretinopathy/'+ image)
                print('moved to noretinopathy')
            except FileNotFoundError:
                print('did not read file')
    