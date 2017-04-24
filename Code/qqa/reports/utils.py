def convert_ndarray_to_list(array):
    l = []
    for x in array:
        l.append(list())
        for y in x:
            l[-1].append(y)
    return l