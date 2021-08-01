import pickle
import numpy as np
import json

dates = []
times = []
freshps = []
rottenps = []

# Read pickle file
with open('/Users/mac/Desktop/oakd/depthai-python/OAK_Course_Examples/data.pickle', "rb") as f:
    loaded_obj = pickle.load(f)

# Generate listes of elements to process
for i in range(len(loaded_obj["record"])):
    dates.append(loaded_obj["record"][i]["date"])
    times.append(loaded_obj["record"][i]["time"])
    freshps.append(loaded_obj["record"][i]['Fresh_percentage'])
    rottenps.append(loaded_obj["record"][i]['Rotten_percentage'])

# Store latest data state
updata = {"date": str(dates[-1]),
          "time": str(times[-1]),
          "fresh_p": np.rint(np.mean(freshps)),
          "rotten_p": np.rint(np.mean(rottenps))
          }

with open("updata.json", "w") as outfile:
    json.dump(updata, outfile)
