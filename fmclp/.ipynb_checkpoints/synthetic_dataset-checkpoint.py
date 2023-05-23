import numpy as np


def synthetic_dataset(size=1000, influence=True):
    """size - number of observations 
       influence - if True, then dependence between sensitive attribute and label is imposed. If False, then the 
                   sensitive attribute is independent of the label.  
    """

    attr = np.random.choice([0, 1], size=size)
    error_x = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_y = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_z = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_target = np.random.normal(loc=0.0, scale=0.5, size=size)

    y1 = np.random.normal(loc=1, scale=1, size=size)
    y2 = np.random.normal(loc=1, scale=1, size=size)
    y3 = np.random.normal(loc=1, scale=1, size=size)

    x = y1 + y2 + error_x
    y = y1 + y3 + error_y
    z = y2 + y3 + error_z

    if influence == True:
        target = x * (1 + 2 * attr) + y * (1 - 0.5 * attr) + z * (1 + 0.5 * attr) + error_target * attr
    if influence == False:
        target = x + y + z + error_target

    target = simple_splitter(target)

    synthetic_df = pd.DataFrame(np.array((x, y, z, attr, target))).T.rename(
        columns={0: 'x', 1: 'y', 2: 'z', 3: 'attr', 4: 'target'})

    return synthetic_df


def simple_splitter(arr):
    arr_unchanged = arr.copy()
    arr = np.sort(np.array(arr))
    l = len(arr)
    n1 = arr[int(l / 3)]
    n2 = arr[int(2 * l / 3)]
    result = []

    for i in range(l):
        if arr_unchanged[i] <= n1:
            result.append(0)
        elif (arr_unchanged[i] > n1 and arr_unchanged[i] <= n2):
            result.append(1)
        else:
            result.append(2)

    return np.array(result)
