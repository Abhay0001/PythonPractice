dict1 = {
    "CURVE_1": "CURVE_2", "CURVE_2": "CURVE_3", "CURVE_4": "CURVE_3", "CURVE_5": "CURVE_1",
    "CURVE_10": "CURVE_20", "CURVE_30": "CURVE_20"}
dict_old = dict1.copy()

for key, value in dict_old.items():
    print(f"{key}: {value}")

dict_new = {}

for key in dict_old:
    iter = dict_old[key]
    while iter in dict_old:
        iter = dict_old[iter]
    dict_new[key] = iter

print(dict_new)
