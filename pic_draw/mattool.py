def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    import scipy.io as spio
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
 
def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    import scipy.io as spio
    import numpy

    if isinstance(dict, spio.matlab.mio5_params.mat_struct):
        dict = _todict(dict)
    if isinstance(dict, numpy.number):
        return dict
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
        if isinstance(dict[key], numpy.ndarray):
            dict[key] = _arr_todict(dict[key])
    return dict
 
def _arr_todict(arr):
    new_arr = []
    for i in arr:
        new_arr.append(_check_keys(i))
    return new_arr

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    import scipy.io as spio
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
 
def mat2json(mat_path=None, filepath = None):
    """
    Converts .mat file to .json and writes new file
    Parameters
    ----------
    mat_path: Str
        path/filename .mat存放路径
    filepath: Str
        如果需要保存成json, 添加这一路径. 否则不保存
    Returns
        返回转化的字典
    -------
    None
    Examples
    --------
    >>> mat2json(blah blah)
    """
 
    import os
    import pandas as pd
    matlabFile = loadmat(mat_path)
    #pop all those dumb fields that don't let you jsonize file
    matlabFile.pop('__header__')
    matlabFile.pop('__version__')
    matlabFile.pop('__globals__')
    #jsonize the file - orientation is 'index'
    matlabFile = pd.Series(matlabFile).to_json()
 
    if filepath:
        json_path = os.path.splitext(os.path.split(mat_path)[1])[0] + '.json'
        with open(json_path, 'w') as f:
                f.write(matlabFile)
    return matlabFile