def convert_2darray_to_list(array):
    l = []
    for x in array:
        l.append(convert_1darray_to_list(x))
    return l

def convert_1darray_to_list(array):
    l = []
    for x in array:
        l.append(round(x, 2))
    return l