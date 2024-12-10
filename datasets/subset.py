with open("shakespeare-data.txt", "r") as file:
    data = file.readlines()

subset_data = data[:len(data) // 3]

with open("shakespeare-data-subset.txt", "w") as file:
    file.writelines(subset_data)