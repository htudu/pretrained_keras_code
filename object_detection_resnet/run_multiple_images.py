import argparse
import os
import time
starttime = time.time()
from resnet_50 import ResNET

files = os.listdir("data/")
model = ResNET()
print("Loading Time Taken {}".format(int(time.time() - starttime)))

for fl in files:
    starttime = time.time()
    filename = f"data/{fl}"    
    result = model.run_object_prediction(filename)
    print("\n ====================== \n")
    print("Results : ")
    for itm in result[0]:
        print(f"{itm[1]} - {itm[2]*100}")
    print("Time Taken - {:.2f}".format(float(time.time() - starttime)))
