def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

def read_list_from_file(filename):
    lst = []
    with open(filename, 'r') as file:
        for line in file:
            item = line.strip()
            item_float = float(item)
            lst.append(item_float)
    return lst
