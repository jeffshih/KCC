import cv2
from colorClassifier import colorClassifier 
import random
import glob 
import argparse
from argparse import ArgumentParser





if __name__ == "__main__":


    modelPath = './colorModel1017.h5'

    classifier = colorClassifier.colorClassifier(modelPath)

    parser = ArgumentParser()
    parser.add_argument("-i",'--image',help="input image path",dest="img")

    args = parser.parse_args()

    if args.img == None:
        allimg = glob.glob('../TestSet_01_perColor_500/*/*.jpg')
        label = [i.split("/")[2] for i in allimg]
        for x in range(len(label)):
            if random.random()>0.8:
                inp = cv2.imread(allimg[x])
                res = classifier.getColor(inp)
                print "ground Truth is {} and result is {}".format(label[x],res)

    else:
        inp = cv2.imread(args.img)
        res = classifier.getColor(inp)
        print "result is {}".format(res)
