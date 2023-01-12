import imp
import json
import scipy.io

path="./trash/test.json"
with open(path, "r") as f:
    data = json.load(f)
scipy.io.savemat('test0.mat', mdict=data)