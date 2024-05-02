import numpy as np
from time import time
import coremltools as ct

modelc = ct.models.CompiledMLModel("model.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)

input_ids = np.random.randint(0, 50265, (4, 512), dtype=np.int32)

n = 100
start = time()
for _ in range(n):
    outputs = modelc.predict({
        "input_ids": input_ids
    })["outputs"]
    print(outputs[0, 0, 0, 0])
end = time()

print("time in ms: ", (end - start) * 1000 / n)
print("tokens per sec: ", n / (end - start))