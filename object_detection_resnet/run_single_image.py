import argparse
from resnet_50 import ResNET

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default='banana.jpg')
args = parser.parse_args()

filename = args.file
model = ResNET()

import time
starttime = time.time()
# import tms # This is my module that takes forever to load
result = model.run_object_prediction(filename)
print("\n ====================== \n")
print("Results : ")
for itm in result[0]:
    print(f"{itm[1]} - {itm[2]*100}")
print("Time Taken {}".format(int(time.time() - starttime)))
