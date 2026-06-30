import numpy as np
import struct
import os

TRAIN_IMG = os.path.join("data", "train-images.idx3-ubyte")
TRAIN_LBL = os.path.join("data", "train-labels.idx1-ubyte")
TEST_IMG = os.path.join("data", "t10k-images.idx3-ubyte")
TEST_LBL = os.path.join("data", "t10k-labels.idx1-ubyte")

def load_idx_images(filepath) -> np.ndarray:
    """
    Image loader to read in IDX files to a NumPy array.
        Filepath: .idx3-ubyte file
    """
    with open(filepath, "rb") as f:
        # Read the header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read the remaining pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape into images
        images = data.reshape(num_images, rows, cols)

        return images

def load_idx_labels(filepath) -> np.ndarray:
    """
    Image loader to read in IDX files to a NumPy array.\n\t
        Filepath: .idx1-ubyte file
    """
    with open(filepath, "rb") as f:
        # Read the header
        magic, num_labels = struct.unpack(">II", f.read(8))

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels
    
def transform_vector(v: np.ndarray) -> np.ndarray:
    """
    Transforms the n-d NumPy array of input
    into a 784, 1 input vector.
        
        v: the n-d numpy array of input values
    """
    outer = []
    for p in v:
        inner = []
        for d in p:
            for g in d:
                inner.append(int(g))
        outer.append(np.array(inner))

    return np.array(outer)
        

def one_hot(target: np.ndarray) -> np.ndarray:
    """
    One-hot encodes the truth values for MNIST dataset.
    
    \tx: the integer truth value
    
    Outputs an array where the index represents the truth value.
    """
    x = len(target)

    temp = []
    for i in range(x):
        t = [0 * i for i in range(10)]
        try:
            t[target[i]] += 1
        except IndexError as e:
            print(f"Truth value out of MNIST range:\n\t{e}")
        temp.append(t)
    return np.array(temp)
        
    
if __name__ == "__main__":
    i_train = load_idx_images(TRAIN_IMG)
    l_train = load_idx_labels(TRAIN_LBL)
    i_test = load_idx_images(TEST_IMG)
    l_test = load_idx_labels(TEST_LBL)

    it_array = transform_vector(i_train)
    tt_array = one_hot(l_train)
    
    