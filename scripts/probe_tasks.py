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
        if (
            "kettle" in d
            and "light switch" in d
            and "slide cabinet" in d
            and "microwave" in d
        ):
            # print(i)
        # if (
        #     t1 in d
        #     and t2 in d
        #     and t3 in d
        #     and t4 in d
        # ):
            if len(d) == 4:
                two_tasks = d[0]+' '+d[1]
                # if two_tasks == "kettle light switch":
                #     print("yess === ", i)
                taskset2.add(two_tasks)
                two_tasks = d[2]+' '+d[3]
                if two_tasks == "light switch slide cabinet":
                    print(i)
                taskset2.add(two_tasks)
            else:
                two_tasks = d[0]+' '+d[1]
                taskset2.add(two_tasks)
        else:
            if len(d) == 4:
                two_tasks = d[0]+' '+d[1]
                taskset1.add(two_tasks)
                # if two_tasks == "kettle light switch":
                #     print(i)
                two_tasks = d[2]+' '+d[3]
                # if two_tasks == "kettle light switch":
                #     print(i)
                taskset1.add(two_tasks)
            else:
                two_tasks = d[0]+' '+d[1]
                # if two_tasks == "kettle light switch":
                #     print(i)
                taskset1.add(two_tasks)
        # try:
        #     if d[2] == "hinge cabinet" or d[3]=="hinge cabinet": print(i)
        # except: pass
    # find tasks in taskset1 but not in taskset2
    # print("Unique tasks in taskset1")
    # for task in taskset1:
    #     if task not in taskset2:
    #         print(task)
    # print("Unique tasks in taskset2")
    # for task in taskset2:
    #     if task not in taskset1:
    #         print(combination)
    #         # find count of the task
    #         print(task)
    #         print(taskset1)
    # if "kettle light switch" in two_tasks:
    #     print("kettle light switch")
    #     print(taskset1)
    #     print(taskset2)
    # print(two_tasks)
    break
    # print("--------")
    # print lengths of each set
    # print(len(taskset1))
    # print(len(taskset2))
    # print(taskset1)
    # print(taskset2)
