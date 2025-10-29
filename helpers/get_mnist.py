import os, gzip, urllib.request, numpy as np

URLS = {
    "train_images":"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels":"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images":"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = "./mnist_ubyte"

def _download_if_needed():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in URLS.items():
        out = os.path.join(DATA_DIR, url.split("/")[-1])
        if not os.path.exists(out):
            urllib.request.urlretrieve(url, out)

def _read_images(path_gz):
    with gzip.open(path_gz, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[0:4], "big") == 2051  # magic
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    arr = arr.reshape(n, rows * cols).astype(np.float32) / 255.0
    return arr

def _read_labels(path_gz):
    with gzip.open(path_gz, "rb") as f:
        data = f.read()

    assert int.from_bytes(data[0:4], "big") == 2049  # magic
    n = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    return arr

if __name__ == "__main__":
    _download_if_needed()


# X_train = _read_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
# y_train = _read_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
# X_test  = _read_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
# y_test  = _read_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))