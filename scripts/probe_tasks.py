import json
import numpy as np

file = '/share/portal/kk837/xskill/datasets/kitchen_dataset/task_completions.json'

with open(file) as f:
    task_completions_list = json.load(f)
tasks = ["slide cabinet", "hinge cabinet", "bottom burner", "top burner", "light switch", "microwave", "kettle"]

# find all unique choices of 4 tasks of out of the 7 tasks above 
combinations = []
for i in range(7):
    for j in range(i+1, 7):
        for k in range(j+1, 7):
            for l in range(k+1, 7):
                combinations.append([tasks[i], tasks[j], tasks[k], tasks[l]])

for combination in combinations:
    t1, t2, t3, t4 = combination
    taskset1 = set()
    taskset2 = set()
    # print(combination)
    for i, d in enumerate(task_completions_list):
        # if (
        #     "kettle" in d
        #     and "light switch" in d
        #     and "slide cabinet" in d
        #     and "microwave" in d
        # ):
        if (
            t1 in d
            and t2 in d
            and t3 in d
            and t4 in d
        ):
            if len(d) == 4:
                two_tasks = d[0]+' '+d[1]
                taskset2.add(two_tasks)
                two_tasks = d[2]+' '+d[3]
                taskset2.add(two_tasks)
            else:
                two_tasks = d[0]+' '+d[1]
                taskset2.add(two_tasks)
        else:
            if len(d) == 4:
                two_tasks = d[0]+' '+d[1]
                taskset1.add(two_tasks)
                two_tasks = d[2]+' '+d[3]
                taskset1.add(two_tasks)
            else:
                two_tasks = d[0]+' '+d[1]
                taskset1.add(two_tasks)

    # find tasks in taskset1 but not in taskset2
    # print("Unique tasks in taskset1")
    # for task in taskset1:
    #     if task not in taskset2:
    #         print(task)
    # print("Unique tasks in taskset2")
    for task in taskset2:
        if task not in taskset1:
            print(combination)
            # find count of the task
            print(task)
            print(taskset1)
    # print("--------")
    # print lengths of each set
    # print(len(taskset1))
    # print(len(taskset2))
    # print(taskset1)
    # print(taskset2)


from torch.utils.data import DataLoader, TensorDataset
import torch
from xskill.dataset.dataset import ConcatDataset

# Create two dummy datasets for the example
x1 = torch.randn(100, 10)
y1 = torch.randn(100, 1)
dataset1 = TensorDataset(x1, y1)

x2 = torch.randn(200, 10)
y2 = torch.randn(200, 1)
dataset2 = TensorDataset(x2, y2)

# breakpoint()
# Concatenate the datasets
combined_dataset = ConcatDataset(dataset1, dataset2)
print(len(combined_dataset))
# Create a DataLoader
dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)






