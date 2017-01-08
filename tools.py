import numpy as np

def sort_by_id(X, ids):
    ''' Sort array X by ascending corresponding ids '''
    assert len(X) == len(ids)
    X = np.array(X)
    ids = np.array(ids)
    sorted_args = np.argsort(ids)
    return X[sorted_args], ids[sorted_args]
    
def intersect_sort(X1, ids1, X2, ids2):
    ''' Intersects and sort X1 and X2 based on their respective ids '''
    # Sort both arrays with ids
    X1, ids1 = sort_by_id(X1, ids1)
    X2, ids2 = sort_by_id(X2, ids2)
    
    # Subset arrays by id in common
    # Check that there are no duplicates
    assert len(ids1) == len(set(ids1))
    assert len(ids2) == len(set(ids2))
    common_ids = np.intersect1d(ids1, ids2)
    
    common_args1 = np.in1d(ids1, common_ids)
    X1 = X1[common_args1]
    ids1 = ids1[common_args1]
    
    common_args2 = np.in1d(ids2, common_ids)
    X2 = X2[common_args2]
    ids2 = ids2[common_args2]
    lengths = [item.shape[0] for item in [X1, ids1, X2, ids2]]
    assert all(l == lengths[0] for l in lengths)
    return X1, ids1, X2, ids2

def test_intersect_sort():
    ''' Test intersect sort method '''
    X1 = ['i', 'a', 'b', 'c', 'd', 'e']
    ids1 = [9,1,2,3,4,5]
    X2 = ['G', 'D', 'I', 'C']
    ids2 = [7,4,9,3]
    X1, ids1, X2, ids2 = intersect_sort(X1, ids1, X2, ids2)
    assert (X1 == np.array(['c', 'd', 'i'])).all()
    assert (X2 == np.array(['C', 'D', 'I'])).all()
    assert (ids1 == np.array([3, 4, 9])).all()
    assert (ids2 == np.array([3, 4, 9])).all()
test_intersect_sort()