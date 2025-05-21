import numpy as np
#https://www.geeksforgeeks.org/compute-the-mean-standard-deviation-and-variance-of-a-given-numpy-array/

#ALL

baseline_999 = np.array([0.0, 0.59, 0.83, 0.28, 0.0, 0.0])
badge_999    = np.array([0.0, 0.0, 0.67, 0.81, 0.41, 0.0])
tc7_999      = np.array([0.0, 0.0, 0.67, 0.81, 0.41, 0.0])
full_bl_999  = 0.6024844720496895
full_badge_999 = 0.6413043478260869

baseline_42 = np.array([0.0, 0.0, 0.73, 0.8, 0.63, 0.0])
badge_42    = np.array([0.0, 0.0, 0.68, 0.84, 0.64, 0.0])
full_bl_42  = 0.6956521739130435
full_badge_42 = 0.717391304347826

baseline_123 = np.array([0.0, 0.67, 0.76, 0.42, 0.0, 0.0])
badge_123    = np.array([0.0, 0.0, 0.75, 0.85, 0.6, 0.0])
full_bl_123  = 0.6149068322981367
full_badge_123 = 0.7267080745341615

baseline_1616 = np.array([0.0, 0.0, 0.82, 0.73, 0.6, 0.0])
badge_1616    = np.array([0.0, 0.0, 0.75, 0.81, 0.59, 0.0])
full_bl_1616  = 0.6770186335403726
full_badge_1616 = 0.703416149068323






full_baseline_values = [full_bl_42, full_bl_123, full_bl_999, full_bl_1616]
full_AL_values = [full_badge_42, full_badge_123, full_badge_999, full_badge_1616]

full_bl_overall_mean = np.mean(full_baseline_values)
full_bl_overall_std  = np.std(full_baseline_values)
full_badge_overall_mean = np.mean(full_AL_values)
full_badge_overall_std  = np.std(full_AL_values)



baseline_data = [baseline_42, baseline_123, baseline_999, baseline_1616]
badge_data    = [badge_42, badge_123, badge_999, badge_1616]

baseline_label_acc_mean = []
baseline_label_acc_std = []
badge_label_acc_mean = []
badge_label_acc_std = []

for i in range(len(baseline_42)):
    baseline_values = [arr[i] for arr in baseline_data]
    baseline_label_acc_mean.append(np.mean(baseline_values))
    baseline_label_acc_std.append(np.std(baseline_values))
    
    badge_values = [arr[i] for arr in badge_data]
    badge_label_acc_mean.append(np.mean(badge_values))
    badge_label_acc_std.append(np.std(badge_values))




print("..............ALL\n")

print("Baseline Label Acc Mean:" , baseline_label_acc_mean)
print("Baseline Label Acc Std:" , baseline_label_acc_std)
print("Badge Label Acc Mean:" , badge_label_acc_mean)
print("Badge Label Acc Std:" , badge_label_acc_std)

print("\n \n Full_BL:")
print("Mean=", full_bl_overall_mean)
print("Std=", full_bl_overall_std)
print("Full_AL:")
print("Mean =", full_badge_overall_mean)
print("Std =", full_badge_overall_std)
