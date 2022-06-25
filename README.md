# Neural-Image-Compression-using-BAR

There is no need for GPU to train.<br>
This is draft README.md, it might have insufficient information.<br>
If you want to get more info, please open the [issues](https://github.com/jeju-ticket/Neural-Image-Compression-using-BAR/issues).


## Installation
**Step 1** Download this repo.
```
git clone https://github.com/jeju-ticket/Neural-Image-Compression-using-BAR.git
```

**Step 2** Install the requirements.
I recommend to install this requirements on your virtual environment.
```
pip install -r requirements.txt
```



## Usage
### Anchor (Image based)
```
python anchor.py
```
### Block-based Compression
```
python block_based.py --block 128    # for 2N==128 (We adopted)
python block_based.py --block 128    # for 2N==256
```

### BAR (Block based Adaptive Resizing) (Proposed)
```
python BAR_ver3.py --block 128    # for 2N==128 (We adopted)
python BAR_ver3.py --block 128    # for 2N==256
```
