## Real Device Test

1. Before testing, you must set your IBM API TOKEN in `real-device.py`.

Example:

```bash
python3 real-device.py \
    -dataset demo-dataset-01.csv \
    -s0 16 \
    -lowest 8 \
    -shots_times=2 \
    -P_V 1 \
    -REAL 1 \
    -PATH test
```