with open("./datasets/jul-data.txt", "r") as file:
    data = file.readlines()

subset_data = data[:len(data) // 10]

with open("./datasets/jul-data-subset.txt", "w") as file:
    file.writelines(subset_data)