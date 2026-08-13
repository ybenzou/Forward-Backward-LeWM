# Deprecated upload location (fallback only)

**Preferred shared data root is LeWM/data** (same place as PushT):

```text
/home/yuanben/WorldModel/LeWM/data/tworoom.tar.zst
/home/yuanben/WorldModel/LeWM/data/cube_single_expert.tar.zst
```

This `FBLeWM/data/incoming/` directory is only a **fallback** if you already
dropped files here. The pipeline checks `LOCAL_DATASET_DIR` (`LeWM/data`) first.
