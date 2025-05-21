import numpy as np

#3cat3neurons

baseline_42_3cat = np.array([0.82, 0.85, 0.65])
badge_42_3cat    = np.array([0.69, 0.82, 0.6])
full_bl_42_3cat  = 0.7857142857142857
full_badge_42_3cat = 0.7352941176470589

baseline_123_3cat = np.array([0.67, 0.77, 0.18])
badge_123_3cat    = np.array([0.68, 0.89, 0.27])
full_bl_123_3cat  = 0.5974025974025974
full_badge_123_3cat = 0.6764705882352942

baseline_999_3cat = np.array([0.62, 0.82, 0.32])
badge_999_3cat    = np.array([0.73, 0.84, 0.35])
full_bl_999_3cat  = 0.6623376623376623
full_badge_999_3cat = 0.6797385620915033

baseline_1616_3cat = np.array([0.78, 0.75, 0.54])
badge_1616_3cat    = np.array([0.74, 0.84, 0.65])
full_bl_1616_3cat  = 0.7077922077922078
full_badge_1616_3cat = 0.7663398692810458


full_baseline_values = [full_bl_42_3cat, full_bl_123_3cat, full_bl_999_3cat, full_bl_1616_3cat]
full_AL_values = [full_badge_42_3cat, full_badge_123_3cat, full_badge_999_3cat, full_badge_1616_3cat]

full_bl_overall_mean = np.mean(full_baseline_values)
full_bl_overall_std  = np.std(full_baseline_values)
full_badge_overall_mean = np.mean(full_AL_values)
full_badge_overall_std  = np.std(full_AL_values)



baseline_data = [baseline_42_3cat, baseline_123_3cat, baseline_999_3cat, baseline_1616_3cat]
badge_data    = [badge_42_3cat, badge_123_3cat, badge_999_3cat, badge_1616_3cat]

baseline_label_acc_mean = []
baseline_label_acc_std = []
badge_label_acc_mean = []
badge_label_acc_std = []

for i in range(len(baseline_42_3cat)):
    baseline_values = [arr[i] for arr in baseline_data]
    baseline_label_acc_mean.append(np.mean(baseline_values))
    baseline_label_acc_std.append(np.std(baseline_values))
    
    badge_values = [arr[i] for arr in badge_data]
    badge_label_acc_mean.append(np.mean(badge_values))
    badge_label_acc_std.append(np.std(badge_values))




print("............THREE CATEGORIES \n")

print("Baseline Label Acc Mean_3cat-3neurons:" , baseline_label_acc_mean)
print("Baseline Label Acc Std:" ,baseline_label_acc_std)
print("Badge Label Acc Mean:" , badge_label_acc_mean)
print("Badge Label Acc Std:" , badge_label_acc_std)

print("\n \n Full_BL:")
print("Mean=", full_bl_overall_mean)
print("Std=", full_bl_overall_std)
print("Full_AL:")
print("Mean =", full_badge_overall_mean)
print("Std =", full_badge_overall_std)
